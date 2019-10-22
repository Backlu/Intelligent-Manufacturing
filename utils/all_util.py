#!/usr/bin/env python
# coding: utf-8

# Copyright © 2019 Hsu Shih-Chieh

import matplotlib
import matplotlib.pyplot as plt
def set_font_cn():
    fp = matplotlib.font_manager.FontProperties(fname = 'utils/fonts/NotoSansMonoCJKtc-Regular.otf')
    font_entry = matplotlib.font_manager.FontEntry(fp.get_file(), name=fp.get_name(),
                                                   style=fp.get_style(), variant=fp.get_variant(),
                                                  weight=fp.get_weight(), stretch=fp.get_stretch(), size=fp.get_size())
    matplotlib.font_manager.fontManager.ttffiles.append(fp.get_file())
    matplotlib.font_manager.fontManager.ttflist.append(font_entry)
    plt.rcParams['font.family'] = fp.get_name()