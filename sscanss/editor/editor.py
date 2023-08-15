"""
Class for JSON text editor
"""
from PyQt6 import QtGui
from PyQt6.Qsci import QsciScintilla, QsciLexerJSON, QsciAPIs
from sscanss.editor.autocomplete import instrument_autocompletions

brace_match_chars = {"{": "}", "[": "]", "(": ")", "'": "'", '"': '"'}


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

        self.api = QsciAPIs(self.lexer)
        for keywords in instrument_autocompletions:
            for keyword in keywords:
                descriptor = f'TYPE={keyword.value.Type}, OPTIONAL={keyword.value.Optional}, DESCRIPTION={keyword.value.Description}'
                self.api.add(f"{keyword.value.Key} - {descriptor}")
        self.api.prepare()

        self.setAutoCompletionThreshold(1)
        self.setAutoCompletionCaseSensitivity(False)
        self.setAutoCompletionSource(self.AutoCompletionSource.AcsAPIs)

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

    def keyPressEvent(self, event):
        """On key press, perform autobrace matching"""
        if event.text() in brace_match_chars.keys():
            init_pos = self.cursor().pos()
            if self.selectedText():
                new_text = self.selectedText() + brace_match_chars[event.text()]
                self.removeSelectedText()
            else:
                new_text = brace_match_chars[event.text()]
            self.insert(new_text)
            self.cursor().setPos(init_pos)
        super().keyPressEvent(event)
