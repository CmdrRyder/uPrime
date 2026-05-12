from collections import OrderedDict
import matplotlib.pyplot as plt

_cases = OrderedDict()

_TAB10 = [plt.get_cmap("tab10")(i) for i in range(10)]
_TAB10_HEX = [
    "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
    for r, g, b, _ in _TAB10
]


def next_color():
    used = {v["color"] for v in _cases.values()}
    for color in _TAB10_HEX:
        if color not in used:
            return color
    # All 10 used — wrap around from the start
    return _TAB10_HEX[len(_cases) % 10]


def add_case(name, data, source, color=None, linestyle="-"):
    _cases[name] = {
        "data":      data,
        "color":     color if color is not None else next_color(),
        "linestyle": linestyle,
        "source":    source,
    }


def remove_case(name):
    del _cases[name]


def rename_case(old_name, new_name):
    _cases[new_name] = _cases.pop(old_name)


def get_cases():
    return _cases.copy()


def clear():
    _cases.clear()
