"""
Class for JSON text editor
"""
from PyQt6 import QtGui
from PyQt6.Qsci import QsciScintilla, QsciLexerJSON
from sscanss.config import settings


class Editor(QsciScintilla):
    """Creates a QScintilla text editor with JSON Lexer

    :param parent: main window instance
    :type parent: MainWindow
    """
    def __init__(self, parent):

        super().__init__(parent)

        font_family = settings.value(settings.Key.Editor_Font_Family)
        font_size = settings.value(settings.Key.Editor_Font_Size)

        self.updateStyle(font_family,font_size)

    def updateStyle(self,family,size):
        """Updates the editor style. Caches currently selected font family and font size to the global settings.
            :param family: font family as string
            :type family: str
            :param size: font size as integer
            :type size: int
            """
        settings.setValue(settings.Key.Editor_Font_Family,family)
        settings.setValue(settings.Key.Editor_Font_Size,size)

        font = QtGui.QFont()
        font.setFamily(family)
        font.setFixedPitch(True)
        font.setPointSize(size)
        self.setFont(font)
        self.setMarginsFont(font)
        font_metrics = QtGui.QFontMetrics(font)
        self.setMarginsFont(font)
        self.setMarginWidth(0, font_metrics.horizontalAdvance("00000") + 6)
        self.setMarginLineNumbers(0, True)
        self.setMarginsBackgroundColor(QtGui.QColor("#cccccc"))
        self.setBraceMatching(QsciScintilla.BraceMatch.SloppyBraceMatch)
        self.setCaretLineVisible(True)
        self.setCaretLineBackgroundColor(QtGui.QColor("#ffe4e4"))

        lexer = QsciLexerJSON()
        lexer.setDefaultFont(font)
        self.setLexer(lexer)
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
