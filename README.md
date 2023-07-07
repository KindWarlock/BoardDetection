# BoardDetection
Обнаружение платы и отлеживание состояния ее светодиодов.

Демонстрация: https://drive.google.com/file/d/1paoGIuwJXPT-KIRSqP-60Fzz21ZtNTHw/view?usp=sharing

---------
Жирный текст является определяющим для участка кода в программе.

## Алгоритм работы

__1.1__ После включения приложения идет получение списка подключенных камер, выбирается последний индекс (при наличии встроенной и внешней в приоритет ставится вторая).

__1.2__ Инициализируются необходимые в работе переменные.

__1.3__ Запускается цикл считывания изображения с камеры до тех пор, пока она или приложение не будут отключены (сюда же входит и нажатие клавиши для выхода из цикла (__1.13__)).

__1.4__ Производится изменение размера изображения, потому что сильно большие для данной задачи излишни.

__1.5__ При нажатии на клавишу "b" запускается поиск платы.

__1.6__ После нахождения происходит поворот кадра, чтобы стороны платы были параллельны сторонам кадра. Параметр `calc_degree` показывающий, нужно ли высчитывать угол поворота, в данном случае имеет значение `True`, поскольку это первая встреча приложения с данной платой. 

__1.7__ После поворота кадр обновился, искать плату нужно заново.

Следующие три пункта срабатывают, если уже была обнаружена плата.

- __1.8__ Кадр обновился, теперь он снова стоит прямо. Это исправляется поворотом, однако теперь уже без пересчета угла.

- __1.9__ Запускается проверка горения светодиодов, находящихся на плате.
- __1.10__ Отрисовывается прямоугольник платы и светодиодов.

Если же платы нет.
- __1.11__ Перерисовываются примерные границы, в которых должна находиться плата.

__1.12__ Кадр со всем нарисованным выводится на экран.
