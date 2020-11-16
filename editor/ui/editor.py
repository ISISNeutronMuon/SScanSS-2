from PyQt5 import QtGui
from PyQt5.Qsci import QsciScintilla, QsciLexerJSON


class Editor(QsciScintilla):
    ARROW_MARKER_NUM = 8

    def __init__(self, parent=None):
        super().__init__(parent)

        # Set the default font
        font = QtGui.QFont()
        font.setFamily('Courier')
        font.setFixedPitch(True)
        font.setPointSize(10)
        self.setFont(font)
        self.setMarginsFont(font)

        # Margin 0 is used for line numbers
        font_metrics = QtGui.QFontMetrics(font)
        self.setMarginsFont(font)
        self.setMarginWidth(0, font_metrics.width("00000") + 6)
        self.setMarginLineNumbers(0, True)
        self.setMarginsBackgroundColor(QtGui.QColor("#cccccc"))

        # Brace matching: enable for a brace immediately before or after
        # the current position
        #
        self.setBraceMatching(QsciScintilla.SloppyBraceMatch)

        # Current line visible with special background color
        self.setCaretLineVisible(True)
        self.setCaretLineBackgroundColor(QtGui.QColor("#ffe4e4"))

        lexer = QsciLexerJSON()
        lexer.setDefaultFont(font)
        self.setLexer(lexer)
        self.SendScintilla(QsciScintilla.SCI_STYLESETFONT, 1, b'Courier')

        # Don't want to see the horizontal scrollbar at all
        # Use raw message to Scintilla here (all messages are documented
        # here: http://www.scintilla.org/ScintillaDoc.html)
        # self.SendScintilla(QsciScintilla.SCI_SETHSCROLLBAR, 0)

        # not too small
        self.setScrollWidth(1)
        self.setEolMode(QsciScintilla.EolUnix)
        self.setScrollWidthTracking(True)
        self.setMinimumSize(500, 600)
        self.setFolding(True)
        self.setIndentationsUseTabs(False)
        self.setIndentationGuides(True)
        self.setAutoIndent(True)
        self.setTabWidth(4)
