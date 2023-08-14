"""
Class for JSON text editor
"""
from PyQt6 import QtGui
from PyQt6.Qsci import QsciScintilla, QsciLexerJSON


class Editor(QsciScintilla):
    """Creates a QScintilla text editor with JSON Lexer

    :param parent: main window instance
    :type parent: MainWindow
    """
    def __init__(self, parent):

        super().__init__(parent)

        self.parent = parent
        self.setMarginLineNumbers(0, True)
        self.setMarginsBackgroundColor(QtGui.QColor("#cccccc"))
        self.setBraceMatching(QsciScintilla.BraceMatch.SloppyBraceMatch)
        self.setCaretLineVisible(True)
        self.setCaretLineBackgroundColor(QtGui.QColor("#ffe4e4"))

        self.lexer = QsciLexerJSON()
        self.setLexer(self.lexer)
        self.SendScintilla(QsciScintilla.SCI_STYLESETFONT, 1, b'Courier')

        self.setScrollWidth(1)
        self.setEolMode(QsciScintilla.EolMode.EolUnix)
        self.setScrollWidthTracking(True)
        self.setMinimumSize(200, 200)
        self.setFolding(QsciScintilla.FoldStyle.PlainFoldStyle)
        self.setIndentationsUseTabs(False)
        self.setIndentationGuides(True)
        self.setAutoIndent(True)
        self.setTabWidth(4)
        self.updateFont()

    def updateFont(self):
        """Updates the editor font"""
        font = QtGui.QFont()
        font.setFamily(self.parent.editor_font_family)
        font.setFixedPitch(True)
        font.setPointSize(self.parent.editor_font_size)
        self.lexer.setFont(font)
        self.setMarginsFont(font)
        self.setMarginWidth(0, QtGui.QFontMetrics(font).horizontalAdvance("00000") + 6)
