"""
A module for working with the board 'Уточка' and similar.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def preprocess(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(9,9),0)
    _, th = cv2.threshold(blur,170,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    morph_kernel = np.ones((11,11))
    enclosed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, morph_kernel)
    return enclosed


class Board:
    """
    Класс платы.

    Поля
    --------
    x : int
        X-координата описывающего прямоугольника.
    y : int
        Y-координата описывающего прямоугольника.
    w : int
        Ширина описывающего прямоугольника.
    h : int
        Высота описывающего прямоугольника.
    working : bool
        Показывает, работают ли ВСЕ светодиоды на плате.
    degree : float
        Угол, на который повернута плата относительно кадра.
    elems : list
        Список элементов (объектов класса LED).
    
    Методы
    ------ 
    find_board(frame: numpy.ndarray, borders: tuple[int, int, int, int]) -> bool
        Ищет плату в прямоугольнике. Возвращает True, если плата найдена, False - в противном случае.
    get_rect() -> tuple[int, int, int, int]
        Получает координаты платы в кортеже (x, y, ширина, высота).
    set_rect(tuple[int, int, int, int])
        Устанавливает значения координат платы по заданному кортежу (x, y, ширина, высота).
    rotate(frame: numpy.ndarray)
        Повернуть изображение, чтобы описывающий прямоугольник маскимально совпадал с границами платы.
    draw(frame: numpy.ndarray)
        Рисует на изображении прямоугольники платы и светодиодов.
    check_lights(frame: numpy.ndarray)
        Проверяет статус светодиодов.
    """

    def __init__(self, rect=(0, 0, 0, 0)):
        """
        Параметры
        ---------
        rect : tuple, optional
            Кортеж, содержащий координаты платы (x, y, width, height).
        """

        self.x = rect[0]
        self.y = rect[1]
        self.w = rect[2]
        self.h = rect[3]
        self.working = False
        # self.color = np.empty((2, 3))
        self._init_elems()
        self.degree = 0


    def _init_elems(self):
        """
        Инициализирует светодиоды.

        Размеры получены из отношения реального размера элемента к реальному размеру платы. 
        """

        self.elems = []
        # ------- WHITE LED --------
        led_l_ratio = 3.4 / 10
        led_w_ratio = 0.8 / 10
        led_h_ratio = 0.4 / 12
        led1_t_ratio = 1.8 / 12
        led_space_ratio = 0.5 / 12
        # ------ INDICATORS --------
        ind_l_ratio = 7.7 / 10
        ind_w_ratio = 0.4 / 10
        ind_h_ratio = 0.4 / 12
        ind1_t_ratio = 8.4 / 12
        ind_space_ratio = 0.4 / 12

        colors = ['w', 'w', 'w', 'r', 'b', 'y', 'g']
        # В разных версиях платы количество светодиодов различается, поэтому надо узнать индекс последнего.
        last_w = max(loc for loc, val in enumerate(colors) if val == 'w')

        for i in range(7):
            color = colors[i]

            l_ratio = led_l_ratio if i <= last_w else ind_l_ratio
            w_ratio = led_w_ratio if i <= last_w else ind_w_ratio
            h_ratio = led_h_ratio if i <= last_w else ind_h_ratio
            t1_ratio = led1_t_ratio if i <= last_w else ind1_t_ratio
            space_ratio = led_space_ratio if i <= last_w else ind_space_ratio
            
            # Цветные и белые светодиоды расположены в разных метах, так что разделим индексы
            idx = i if i <= last_w else i - last_w - 1

            # Белых светодиодов несколько, поэтому чтобы их различать, добавим порядковые номера.
            name = f'{colors[i]}{i}' if i <= last_w else colors[i]

            w = w_ratio * self.w 
            x = l_ratio * self.w + self.x
            h = h_ratio * self.h 
            y = (t1_ratio + (h_ratio + space_ratio) * idx) * self.h + self.y
            rect = np.array([x, y, w, h], dtype='uint')
            led = Led(rect, color, name)
            self.elems.append(led)


    def find_board(self, frame: np.ndarray, borders: tuple[int, int, int, int]) -> bool:
        """
        Ищет плату в прямоугольнике. Возвращает True, если плата найдена, False - в противном случае.

        Параметры
        ---------
        frame : numpy.ndarray
            Необрезанное RGB-изображение с платой.
        borders : tuple[int, int, int, int]
            Рамки, в которых искать плату.

        Возвращает
        ----------
        bool
            True, если плата найдена, False - если нет.
        """

        # Находим медианный цвет.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        median_color = np.median(hsv[borders[1]:borders[1] + borders[3], borders[0]:borders[0] + borders[2], :], axis=(0, 1))
        # Границы зеленого цвета.
        if median_color[0] not in range(32, 95):
            return False, None

        color = np.empty((2, 3))
        # Определяем верхнюю и нижнюю границы цвета.
        color[0, [0, 1, 2]] = median_color[[0, 1, 2]] - [5, 40, 50]
        color[1, [0, 1, 2]] = median_color[[0, 1, 2]] + [5, 40, 50]
        # print(color)
        
        # Получаем HSV-маску.
        mask = np.zeros((hsv.shape[0], hsv.shape[1]), dtype = "uint8")
        mask += cv2.inRange(hsv, color[0], color[1])
        
        # Накладываем HSV-маску.
        masked = cv2.bitwise_and(frame, frame, mask=mask)
        # Получаем бинарное изображение.
        processed_frame = preprocess(masked)
    
        # Находим наибольший прямоугольник. Его и будем считать за искомый.
        contours, hierarchy = cv2.findContours(processed_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rect = []
        max_size = 0
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h > max_size:
                max_size = w * h
                rect = (x, y, w, h)

        self.set_rect(rect)

        return True


    def _find_connector(self, frame: np.ndarray) -> tuple[int, int, int, int, np.ndarray]:
        """
        Находит и возвращает координаты сетевого разъема.

        Поиск происходит в предопределенных границах. Используется для определеня угла поворота.

        Параметры
        ---------
        frame : numpy.ndarray
            Необрезанное RGB-изображение с платой.

        Возвращает 
        ----------
        x : int
            X-координата описывающего прямоугольника.
        y : int
            Y-координата описывающего прямоугольника.
        width : int
            Ширина описывающего прямоугольника.
        height : int
            Высота описывающего прямоугольника.
        edges_full : numpy.ndarray
            Состоящий из нулей массив размером с исходное изображения, содержащий найденный контур.
        """

        # Определяем границы участка, в котором будет производиться поиск.
        x = int(1.5 / 10 * self.w) + self.x
        w = int(6 / 10 * self.w)
        y = int(7.5 / 12 * self.h) + self.y
        h = int(3.8 / 12 * self.h)
        img = frame[y:y + h, x:x + w]

        # Поиск контуров
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist(img, [0], None, [256], [0,256])

        plt.plot(hist[1:])
        # plt.show()
        # TODO изменить пороги в зависимости от значений. Или проводить нормализацию. 
        edges = cv2.Canny(image=gray, threshold1=80, threshold2=100)
        # Выделение контуров 
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)
        
        # Поиск контура с максимальной площадью
        rect = (0, 0, 0, 0)
        max_size = 0
        for c in contours:
            c_x, c_y, c_w, c_h = cv2.boundingRect(c)
            if c_w * c_h > max_size:
                max_size = c_w * c_h
                max_c = c
                rect = (x + c_x, y + c_y, c_w, c_h)
        if max_size == 0:
            # TODO error
            pass

        # Подходтовка результирующего массива.
        edges_full = np.zeros((frame.shape[0], frame.shape[1]))

        # Выделение контура разъема.
        cv2.drawContours(edges_full[y:y + h, x:x + w], max_c, -1, 1, 1)
        # cv2.imshow("edges", edges_full)
        return rect, edges_full


    def get_rect(self) -> tuple[int, int, int, int]:
        """
        Получает координаты платы в кортеже (x, y, ширина, высота).

        Возвращает 
        ----------
        x : int
            X-координата описывающего прямоугольника.
        y : int
            Y-координата описывающего прямоугольника.
        w : int
            Ширина описывающего прямоугольника.
        h : int
            Высота описывающего прямоугольника.
        """

        return (self.x, self.y, self.w, self.h)
    
    
    def set_rect(self, rect: tuple[int, int, int, int]):
        """
        Устанавливает значения координат платы по заданному кортежу (x, y, ширина, высота).

        Параметры
        ---------
        x : int
            X-координата описывающего прямоугольника.
        y : int
            Y-координата описывающего прямоугольника.
        w : int
            Ширина описывающего прямоугольника.
        h : int
            Высота описывающего прямоугольника.
        """

        self.x = rect[0]
        self.y = rect[1]
        self.w = rect[2]
        self.h = rect[3]

        # После изменения прямоугольника относительные расстояния тоже надо пересчитать
        self._init_elems()
        
        # На проверенное плате такое не проводится, поэтому сбрасываем значение к дефолтному
        self.working = False


    def rotate(self, frame: np.ndarray, calc_degree=False):
        """
        Повернуть изображение, чтобы описательный прямоугольник маскимально совпадал с границами платы.

        Параметры
        ---------
        frame : numpy.ndarray
            Необрезанное RGB-изораажение с платой.
        calc_degree : bool, optional
            Если имеет значение True, высчитвается угол, иначе берется из свойства.
        """
        if calc_degree:
            self._get_rotation_degree(frame)
        # Поворот происходит вокруг центра изображения
        # TODO вокруг платы
        frame_center = tuple(np.array(frame.shape[1::-1]) / 2)
        m = cv2.getRotationMatrix2D(frame_center, self.degree, 1)
        cv2.warpAffine(frame, m, frame.shape[1::-1], dst=frame, flags=cv2.INTER_LINEAR)
        
        

    def _get_rotation_degree(self, frame: np.ndarray):
        """
        Получить угол поворота платы.

        Параметры
        ---------
        frame : numpy.ndarray
            Необрезанное RGB-изображение с платой.
        """

        # Как ориентир используем разъем вместо всей платы для получения более четких границ.
        (x, y, w, h), connector_edges = self._find_connector(frame)
        cropped = connector_edges[y:y + h, x:x + w]
        mask_x = cropped[0, :] != 0
        mask_y = cropped[:, 0] != 0
        
        # Ищем угол поворота по arctg. Для этого берем левый верхний треугольник и угол, 
        # противолежащий к горизонтальному катету.
        opposite_side = mask_x.argmax(axis=0)

        # Работая с небольшими углами на небольшом разрешении, возникает проблема, что 
        # мы получаем не диагональ, а ломанную, состоящую из нескольких горизонтальных или 
        # (как в данном случае) вертикальных ступеней, из-за чего более правильным будет искать крайнее 
        # значение с другого конца и отнимать его от всей длины.
        belonging_side = cropped.shape[0] - np.flip(mask_y).argmax(axis=0)
        
        # Учитывая, что отклонения платы меньше 45 градусов, из случая, когда противолежащий катет 
        # больше прилежащего, выводы следующие:
        # 1) Небольшим теперь считается горизонтальный угол, поэтому надо пересчитать катеты.
        # 2) Поворот происходит в другую сторону. 
        # 3) Отношение горизонтального катета к вертикальному теперь являются аргументом для arcctg.
        if belonging_side < opposite_side:
            opposite_side =  cropped.shape[1] - np.flip(mask_x).argmax(axis=0)
            belonging_side = mask_y.argmax(axis=0)
            rad = np.arctan(opposite_side / belonging_side)
            if np.isnan(rad):
                return 0
            self.degree = np.rad2deg(rad) - 90
        else:
            rad = np.arctan(opposite_side / belonging_side)
            if np.isnan(rad):
                return 0
            self.degree = np.rad2deg(rad)


    def draw(self, frame: np.ndarray):
        """
        Рисует на изображении прямоугольники платы и светодиодов.

        Прямоугольники имеют фиолетовую рамку, если элемент не работает, зеленую 
        в ином случае 

        Параметры
        ---------
        frame : numpy.ndarray
            Необрезанное RGB-изораажение с платой.
        """

        board_color = (143, 217, 74) if self.working else (219, 68, 126)
        cv2.rectangle(frame, self.get_rect(), board_color, 3)
        for e in self.elems:
            led_color = (143, 217, 74) if e.working else (219, 68, 126)
            cv2.rectangle(frame, e.get_rect(), led_color, 2)
    

    def check_lights(self, frame: np.ndarray):
        """
        Проверяет статус светодиодов.

        Если все светодиоды работают, плата тоже принимает статус рабочей.

        Параметры
        ---------
        frame : numpy.ndarray
            Необрезанное RGB-изораажение с платой.
        """

        if self.working:
            return
        for e in self.elems:
            e.check_lights(frame)
        
        # Если хоть один элемент не работает (имеет значение False, т.е. 0), то
        # произведение будет равно 0, и плата не будет считаться рабочей.
        if np.prod([e.working for e in self.elems]) == 1:
            self.working = True
            print("The board is fully working!")


class Led:
    """
    Класс светодиода.

    Поля
    ----
    x : int
        X-координата вписанного прямоугольника.
    y : int
        Y-координата вписанного прямоугольника.
    w : int
        Ширина вписанного прямоугольника.
    h : int
        Высота вписанного прямоугольника.
    working : bool
        Показывает, был ли светодиод зажжен хотя бы раз.
    color : str
        Буква, дающая указание на цвет
    name : str
        Уникальное имя для светодиода. Состоит из цвета и, для белых, индекса.
    
    Методы
    ------ 
    color_hue() -> numpy.ndarray((2, 2))
        Получить верхнюю и нижнюю границу оттенка (0..180) в зависимости от цвета.
    get_rect() -> int, int, int, int
        Получить координаты в кортеже (x, y, width, height).
    check_lights()
        Проверить статус светодиода.
    """

    def __init__(self, rect: tuple[int, int, int, int], color: str, name: str):
        """
        Параметры
        ---------
        rect : tuple
            Кортеж с координатами светодиода (x, y, width, height).
        color : str
            Буква, дающая указание на цвет.
        name : str
            Отличительное имя светодиода. Состоит из цвета и, для белых, индекса.
        """

        self.x = rect[0]
        self.y = rect[1]
        self.w = rect[2]
        self.h = rect[3]
        self.working = False
        self.color = color
        self.name = name


    def color_hue(self) -> np.ndarray:
        """
        Получить верхнюю и нижнюю границу оттенка (0..180) в зависимости от цвета.
        
        Значения получены эмпирическим путем, от соотношения цвета и его HSV значения.
        
        Возвращает
        ----------
        hue : numpy.ndarray((2, 2))
            hue[0] - нижняя граница.
            hue[1] - верхняя граница.
            hue[2] - нижняя граница второго возможного отрезка цвета.
            hue[3] - верхняя граница второго возможного отрезка цвета.
        """
        
        hue = np.empty((2,2))
        # Красный может находиться либо в начале палитры, либо в конце.
        if self.color == 'r':
            hue[0, 0] = 3
            hue[0, 1] = 25
            hue[1, 0] = 150
            hue[1, 1] = 180
        elif self.color == 'g':
            hue[0, 0] = 40
            hue[0, 1] = 95 
        elif self.color == 'b':
            hue[0, 0] = 98
            hue[0, 1] = 120
        # Отслеживаемый желтый близок к белому.
        elif self.color == 'y':
            hue[0, 0] = 20
            hue[0, 1] = 35 
            hue[1, 0] = 0
            hue[1, 1] = 0
        elif self.color == 'w':
            hue[0, 0] = 0
            hue[0, 1] = 180 
        return hue
    

    def get_rect(self):
        """
        Получить координаты в кортеже (x, y, width, height).

        Возвращает 
        ----------
        x : int
            X-координата вписанного прямоугольника.
        y : int
            Y-координата вписанного прямоугольника.
        width : int
            Ширина вписанного прямоугольника.
        height : int
            Высота вписанного прямоугольника.
        """

        return self.x, self.y, self.w, self.h


    def check_lights(self, frame: np.ndarray):
        """
        Проверить статус светодиодов.

        Светодиод признается рабочим, если его яркость выше определенного порога. То есть
        тусклые светодиоды не подходят (часть тебований).

        Параметры
        ---------
        frame : numpy.ndarray
            Необрезанное RGB-изображение с платой.
        """

        if self.working:
            return
        led_img = np.empty((0, 0))

        # Белые светодиоды светят значительно ярче (их свет занимает бОльшую площадь),
        # к тому же все имеют одинаковый свет, что, в случае измерения исключительно по 
        # измеренным размерам, может давать неверный результат (сложно сказать, в область  
        # попал свет соседнего светодиода или его собственный). Отчего и идет увеличение ширины 
        # отслеживаемого окна.
        if self.color == 'w':
            x_start = int(self.x - self.w * 3)
            x_stop = int(self.x + self.w * 4)
            led_img = frame[self.y:self.y + self.h, x_start:x_stop]
        else:
            led_img = frame[self.y:self.y + self.h, self.x:self.x + self.w]
        hsv = cv2.cvtColor(led_img, cv2.COLOR_BGR2HSV)
        hue_bottom = self.color_hue()[0, 0]
        hue_top = self.color_hue()[0, 1]
        h = np.median(hsv[:, :, 0])
        s = np.median(hsv[:, :, 1])
        v = np.median(hsv[:, :, 2])
        
        if (v > 230) and (s < 30) and (h >= hue_bottom) and (h <= hue_top):
            self.working = True
            print(f"LED {self.name} is working! Median colors are: {h, s, v}")
        # Особые случаи
        elif (self.color == 'y' or self.color == 'r') and (v > 230) and (s < 30):
            self.working = True
            print(f"LED {self.name} is working!")