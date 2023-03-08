|

.. image:: docs/_static/logo/dark.png#gh-dark-mode-only
    :width: 225
    :class: only-dark
    :align: center
    :alt: Logo

.. image:: docs/_static/logo/light.png#gh-light-mode-only
    :width: 225
    :class: only-light
    :align: center
    :alt: Logo

|
|

*The Chess Computer for nerds, by nerds.*


.. image:: https://img.shields.io/discord/1083067212803354624?style=flat-square
    :alt: Discord
    :target: https://discord.com/invite/wdMdBr6jxs

.. image:: https://img.shields.io/github/license/ierror/acid-chess?style=flat-square
   :alt: GitHub

|

Picture by Picture
==================

ACID Chess is a chess computer written in Python, which can be used with any? board. By filming the board, the
contour of the board is recognized, and the positions of the individual pieces can be determined. Two neural networks
were trained for the board and squares recognition.

.. image:: docs/_static/photos/over-the-board.jpg
    :width: 600
    :alt: How it works - over the board

|

Features
========

You can play against an engine, Stockfish or Maia are available, or play a game against another human. In both variants,
a PGN is generated, which you can load later in the analysis board at Lichess, or so, for analysis.

- Engine play against Stockfish or Maia
- Use polyglot opening books
- PGN exports

.. image:: docs/_static/photos/gui.jpg
  :width: 600
  :alt: How it works - GUI

|

Technology
==========

- Python as a programming language
- Qt as toolkit for the GUI (with own extension for reactive bindings)
- PyTorch (Lightning ) for the development of AI models

|

I want to play against ACID!
============================

We have tested ACID Chess with four different boards and were able to complete games without significant flaws. There
will be problems on unknown boards, but every tester makes ACID Chess better!

Regardless of the chosen installation method: ACID Chess saves images of data that cannot be classified sufficiently.
Please provide us with this data. Create an `issue <https://github.com/ierror/acid-chess/issues/new>`_ in github and upload a ZIP file as an attachment. *<3*

There are two ways to install ACID Chess.

1. as binary: for users who want to try ACID Chess and don't want to deal with installing Python etc.
2. check out the project via git and install the dependencies manually for people who want to develop on ACID Chess themselves.


Contact
=======

- Mastodon `@boerni@chaos.social <https://chaos.social/@boerni>`_
- `Discord <https://discord.com/invite/wdMdBr6jxs>`_
