# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
from enum import Enum, auto

_registered_mouse_buttons = dict()
_registered_keys = dict()


class WispMouseButton(Enum):
    LEFT_BUTTON = auto()
    MIDDLE_BUTTON = auto()
    RIGHT_BUTTON = auto()

    @classmethod
    def register_symbol(cls, wisp_mb: WispMouseButton, symbol):
        _registered_mouse_buttons[wisp_mb] = symbol

    def __eq__(self, other):
        """ Perform equality test with the mapped symbol """
        mapped_symbol = _registered_mouse_buttons.get(self, None)
        assert mapped_symbol is not None, f"Error: WispMouseButton {self.name} have not been registered by the app."
        return mapped_symbol == other

    def __hash__(self):
        return hash(self.value)


class WispKey(Enum):
    LEFT = auto()
    RIGHT = auto()
    UP = auto()
    DOWN = auto()
    a = auto()
    b = auto()
    c = auto()
    d = auto()
    e = auto()
    f = auto()
    g = auto()
    h = auto()
    i = auto()
    j = auto()
    k = auto()
    l = auto()
    m = auto()
    n = auto()
    o = auto()
    p = auto()
    q = auto()
    r = auto()
    s = auto()
    t = auto()
    u = auto()
    v = auto()
    w = auto()
    x = auto()
    y = auto()
    z = auto()
    A = auto()
    B = auto()
    C = auto()
    D = auto()
    E = auto()
    F = auto()
    G = auto()
    H = auto()
    I = auto()
    J = auto()
    K = auto()
    L = auto()
    M = auto()
    N = auto()
    O = auto()
    P = auto()
    Q = auto()
    R = auto()
    S = auto()
    T = auto()
    U = auto()
    V = auto()
    W = auto()
    X = auto()
    Y = auto()
    Z = auto()
    ONE = auto()
    TWO = auto()
    THREE = auto()
    FOUR = auto()
    FIVE = auto()
    SIX = auto()
    SEVEN = auto()
    EIGHT = auto()
    NINE = auto()
    ZERO = auto()
    PLUS = auto()
    MINUS = auto()
    LEFT_BRACKET = auto()
    RIGHT_BRACKET = auto()
    LEFT_SQUARE_BRACKET = auto()
    RIGHT_SQUARE_BRACKET = auto()
    LEFT_CURLY_BRACKET = auto()
    RIGHT_CURLY_BRACKET = auto()
    DOT = auto()
    COMMA = auto()

    @classmethod
    def register_symbol(cls, wisp_key: WispKey, symbol):
        _registered_keys[wisp_key] = symbol

    def __eq__(self, other):
        mapped_symbol = _registered_keys.get(self, None)
        assert mapped_symbol is not None, f"Error: WispKey {self.name} have not been registered by the app."
        return mapped_symbol == other

    def __hash__(self):
        return hash(self.value)
