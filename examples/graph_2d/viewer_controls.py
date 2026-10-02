"""Top strip controls, dropdown wiring, and keyboard help dialog for the graph viewer."""

from __future__ import annotations

import tkinter
from collections.abc import Callable
from tkinter import ttk

#: The pointer gestures, which no key table can carry.
MOUSE_HELP = (
    ("left click", "pick a target vertex, or click empty space to query those coordinates"),
    ("right click", "set the traversal start node"),
    ("double click", "run the path query"),
    ("drag", "pan the zoomed view"),
    ("scroll", "zoom towards the cursor"),
)

#: Extra space, in figure pixels, opened between each checkbox frame and its label. Kept small so the
#: label leaves room to the next frame in the row; the same nudge is applied to the 3D toggle.
CHECK_LABEL_GAP_PX = 3


def make_field(
    strip: tkinter.Frame,
    label: str,
    value: str,
    from_: float,
    to: float,
    increment: float,
    commit: Callable[[str], None],
    width: int = 4,
    format: str | None = None,
) -> tuple[tkinter.StringVar, ttk.Spinbox]:
    """Adds one labelled spinbox field to the strip and returns the (StringVar, Spinbox) pair."""
    tkinter.Label(strip, text=label, bg="#f2f4f7", fg="#5f6368", font=("DejaVu Sans", 9, "bold")).pack(
        side="left", padx=(0, 6)
    )
    variable = tkinter.StringVar(strip, value=value)
    spinbox = ttk.Spinbox(
        strip,
        from_=from_,
        to=to,
        increment=increment,
        width=width,
        textvariable=variable,
        command=lambda: commit(variable.get()),
        **({"format": format} if format is not None else {}),
    )
    spinbox.configure(font=("DejaVu Sans", 9), background="#ffffff")
    spinbox.bind("<Return>", lambda _event: commit(variable.get()))
    spinbox.bind("<FocusOut>", lambda _event: commit(variable.get()))
    spinbox.pack(side="left", padx=(0, 18))
    return variable, spinbox


def make_dropdown(
    strip: tkinter.Frame,
    label: str,
    options: list[str],
    chosen: str,
    choose: Callable[[str], None],
) -> tuple[tkinter.StringVar, ttk.Combobox]:
    """Adds one labelled dropdown to the strip and returns the (StringVar, Combobox) pair."""
    tkinter.Label(strip, text=label, bg="#f2f4f7", fg="#5f6368", font=("DejaVu Sans", 9, "bold")).pack(
        side="left", padx=(0, 6)
    )
    variable = tkinter.StringVar(strip, value=chosen)
    combobox = ttk.Combobox(
        strip,
        textvariable=variable,
        values=options,
        state="readonly",
        width=max(len(option) for option in options) + 2,
    )
    combobox.configure(font=("DejaVu Sans", 9))
    combobox.bind("<<ComboboxSelected>>", lambda _event: choose(combobox.get()))
    combobox.pack(side="left", padx=(0, 18))
    return variable, combobox


def wire_dropdown(combo: ttk.Combobox, choose: Callable[[str], None]) -> None:
    """Lets the arrow keys choose from an open dropdown exactly as a click would."""
    listbox = f"{combo}.popdown.f.l"
    wired = [False]

    def on_browse(*_args) -> None:
        try:
            if not int(combo.tk.call("winfo", "ismapped", listbox)):
                return
            selected = combo.tk.call(listbox, "curselection")
            if not selected:
                return
            value = str(combo.tk.call(listbox, "get", selected[0]))
            if value == combo.get():
                return
            combo.set(value)
            choose(value)
        except tkinter.TclError:
            pass

    def wire() -> None:
        if wired[0]:
            return
        try:
            if not int(combo.tk.call("winfo", "exists", listbox)):
                return
            widget_class = combo.tk.call("winfo", "class", listbox)
            combo.tk.call("bind", "tags", listbox, (listbox, widget_class, "all"))
            for sequence in ("<KeyRelease>", "<Motion>"):
                command = f"choose_{sequence.strip('<>')}_{id(combo)}"
                combo.tk.createcommand(command, on_browse)
                combo.tk.call("bind", listbox, sequence, command)
            wired[0] = True
        except tkinter.TclError:
            return

    def schedule(*_args) -> None:
        combo.after_idle(wire)

    wire()
    combo.bind("<Button-1>", schedule, add="+")
    combo.bind("<KeyPress>", schedule, add="+")
