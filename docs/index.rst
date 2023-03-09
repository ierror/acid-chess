|

.. raw:: html

    <p align="center">
        <img src="_static/logo/dark.png#gh-dark-mode-only" class="only-dark" align="center" width="25%" alt="Logo">
    </p>

.. raw:: html

    <p align="center">
        <img src="_static/logo/light.png#gh-light-mode-only" class="only-light" align="center" width="25%" alt="Logo">
    </p>

|
|

*The Chess Computer for nerds, by nerds.*

.. image:: https://img.shields.io/readthedocs/acid-chess/latest?style=flat-square
    :alt: Documentation Status
    :target: http://acid-chess.rtfd.io/

.. image:: https://img.shields.io/discord/1083067212803354624?style=flat-square&logo=discord
    :alt: Discord Status
    :target: https://discord.com/invite/wdMdBr6jxs

.. image:: https://img.shields.io/github/license/ierror/acid-chess?style=flat-square
    :alt: GitHub License Status
    :target: https://github.com/ierror/acid-chess/blob/main/LICENSE

|

Picture by Picture
==================

ACID Chess is a chess computer written in Python, which can be used with any? board. By filming the board, the
contour of the board is recognized, and the positions of the individual pieces can be determined. Two neural networks
were trained for the board and squares recognition.

.. image:: _static/photos/over-the-board.jpg
    :width: 80%
    :alt: How it works - over the board

|

Features
========

You can play against an engine, Stockfish or Maia are available, or play a game against another human. In both variants,
a PGN is generated, which you can load later in the analysis board at Lichess, or so, for analysis.

- Engine play against Stockfish or Maia
- Use polyglot opening books
- PGN exports

.. image:: _static/photos/gui.jpg
  :width: 80%
  :alt: How it works - GUI

|

Planned Features
================

- Clock
- Play on Lichess
- ... see Issues for details

|

Technology
==========

- Python as a programming language
- Qt (PySide6) as toolkit for the GUI (with own extension for reactive bindings)
- PyTorch (Lightning ) for the development of AI models

|

I want to play against ACID!
============================

We have tested ACID Chess with four different boards and were able to complete games without significant flaws. There
will be problems on unknown boards, but every tester makes ACID Chess better!

Regardless of the chosen installation method: ACID Chess saves images of data that cannot be classified sufficiently.
Please provide us with this data. Create an `issue <https://github.com/ierror/acid-chess/issues/new>`_ and upload a ZIP
file as an attachment. *<3*

There are two ways to install ACID Chess.

1. as binary: for users who want to try ACID Chess and don't want to deal with installing Python etc.
2. check out the project via git and install the dependencies manually for people who want to develop on ACID Chess themselves.

|

Known bugs and limitations
==========================
- after switching cameras you will see an "Image capture failed: timed out waiting for a preview frame" error in the logs. Workaroud: Select camara you want to use and restart the app

|

Resources
=========

Documentation
**************
`http://acid-chess.rtfd.io/ <http://acid-chess.rtfd.io/>`_

Sourcecode
**********
`https://github.com/ierror/acid-chess <https://github.com/ierror/acid-chess>`_

|

Contact
=======

- Mastodon `@boerni@chaos.social <https://chaos.social/@boerni>`_
- `Discord <https://discord.com/invite/wdMdBr6jxs>`_
