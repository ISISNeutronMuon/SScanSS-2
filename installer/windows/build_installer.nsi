;--------------------------------
;Include Modern UI

!include "MUI2.nsh"
!include "version.nsh"

;--------------------------------
;General

;Properly display all languages
Unicode true

;Define name of the product
!define PRODUCT "SScanSS 2"

;Name and file
Name "${PRODUCT}"
OutFile "${PRODUCT}-${VERSION}-setup.exe"

;Default installation folder
InstallDir "$PROGRAMFILES64\${PRODUCT}"

;Get installation folder from registry if available
InstallDirRegKey HKCU "Software\${PRODUCT}" ""

;Request application privileges for Windows Vista
RequestExecutionLevel admin
  
;--------------------------------
;Pages for the installer

!define MUI_ICON "install.ico"
!define MUI_UNICON "uninstall.ico"
!define MUI_WELCOMEFINISHPAGE_BITMAP "banner.bmp"
!define MUI_UNWELCOMEFINISHPAGE_BITMAP "banner.bmp"
!define MUI_HEADERIMAGE
!define MUI_HEADERIMAGE_RIGHT
!define MUI_HEADERIMAGE_BITMAP "header.bmp"
!define MUI_HEADERIMAGE_UNBITMAP "header.bmp"
!define MUI_DIRECTORYPAGE_TEXT_DESTINATION "Folder"

!insertmacro MUI_PAGE_WELCOME # simply remove this and other pages if you don't want it
!insertmacro MUI_PAGE_LICENSE "..\bundle\app\LICENSE" # link to an ANSI encoded license file
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_INSTFILES

!define MUI_FINISHPAGE_RUN	$INSTDIR\bin\sscanss.exe
!define MUI_FINISHPAGE_RUN_NOTCHECKED
!define MUI_FINISHPAGE_SHOWREADME ""
!define MUI_FINISHPAGE_SHOWREADME_NOTCHECKED
!define MUI_FINISHPAGE_SHOWREADME_TEXT "Create Desktop Shortcut"
!define MUI_FINISHPAGE_SHOWREADME_FUNCTION create_desktop_shortcut

!insertmacro MUI_PAGE_FINISH

;For the uninstaller
!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

  
;--------------------------------
;Languages

;At start will be searched if the current system language is in this list,
;if not the first language in this list will be chosen as language
!insertmacro MUI_LANGUAGE "English"
  
;--------------------------------
;Installer Section

Section "SScanSS 2" sec0
	SectionIn RO # Just means if in component mode this is locked

	;Set output path to the installation directory.
	SetOutPath $INSTDIR

	;Put the following file in the SetOutPath
	File /r "..\bundle\app\bin"
	File /r "..\bundle\app\instruments"
	File /r "..\bundle\app\static"
	File "..\bundle\app\LICENSE"

	;Store installation folder in registry
	WriteRegStr HKCU "Software\${PRODUCT}" "" $INSTDIR

	;Registry information for add/remove programs
	WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT}" "DisplayName" "${PRODUCT}"
	WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT}" "UninstallString" '"$INSTDIR\uninstall.exe"'
	WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT}" "NoModify" 1
	WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT}" "NoRepair" 1

	;Create optional start menu shortcut for uninstaller and Main component
	CreateDirectory "$SMPROGRAMS\${PRODUCT}"
	CreateShortCut "$SMPROGRAMS\${PRODUCT}\${PRODUCT}.lnk" "$INSTDIR\bin\sscanss.exe" "" "$INSTDIR\bin\sscanss.exe" 0

	;Create uninstaller
	WriteUninstaller "uninstall.exe"

SectionEnd

Section "Examples" sec1
	SetOutPath $INSTDIR

	;Put the following file in the SetOutPath
	File /r "..\..\examples"
SectionEnd
 
Section /o "Instrument Editor" sec2
	SetOutPath $INSTDIR\bin
	SetOverwrite on

	;Put the following file in the SetOutPath
	File /r "..\bundle\editor\*.*"

	;Create optional start menu shortcut for editor
	CreateShortCut "$SMPROGRAMS\${PRODUCT}\editor.lnk" "$INSTDIR\bin\editor.exe" "" "$INSTDIR\bin\editor.exe" 0
SectionEnd

;--------------------------------
;Uninstaller Section
!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
	!insertmacro MUI_DESCRIPTION_TEXT ${sec0} "Install main SScanSS component."
	!insertmacro MUI_DESCRIPTION_TEXT ${sec1} "Install example data used in the tutorials."
	!insertmacro MUI_DESCRIPTION_TEXT ${sec2} "Install developer tool for editing instrument description files."
!insertmacro MUI_FUNCTION_DESCRIPTION_END

;--------------------------------
;Uninstaller Section

Section "Uninstall"
	;Remove all registry keys
	DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT}"
	DeleteRegKey HKCU "Software\${PRODUCT}"

	;Delete the installation directory + all files in it
	RMDir /r "$INSTDIR\*.*"
	RMDir "$INSTDIR"

	;Delete Start Menu Shortcuts
	RMDir /r "$SMPROGRAMS\${PRODUCT}"
	Delete "$DESKTOP\${PRODUCT}.lnk"
SectionEnd

!macro UninstallExisting exitcode uninstcommand
	Push `${uninstcommand}`
	Call UninstallExisting
	Pop ${exitcode}
!macroend

Function UninstallExisting
	Exch $1 ; uninstcommand
	Push $2 ; Uninstaller
	Push $3 ; Len
	StrCpy $3 ""
	StrCpy $2 $1 1
	StrCmp $2 '"' qloop sloop
	sloop:
		StrCpy $2 $1 1 $3
		IntOp $3 $3 + 1
		StrCmp $2 "" +2
		StrCmp $2 ' ' 0 sloop
		IntOp $3 $3 - 1
		Goto run
	qloop:
		StrCmp $3 "" 0 +2
		StrCpy $1 $1 "" 1 ; Remove initial quote
		IntOp $3 $3 + 1
		StrCpy $2 $1 1 $3
		StrCmp $2 "" +2
		StrCmp $2 '"' 0 qloop
	run:
		StrCpy $2 $1 $3 ; Path to uninstaller
		StrCpy $1 161 ; ERROR_BAD_PATHNAME
		GetFullPathName $3 "$2\.." ; $InstDir
		IfFileExists "$2" 0 +4
		ExecWait '"$2" /S _?=$3' $1 ; This assumes the existing uninstaller is a NSIS uninstaller, other uninstallers don't support /S nor _?=
		IntCmp $1 0 "" +2 +2 ; Don't delete the installer if it was aborted
		Delete "$2" ; Delete the uninstaller
		RMDir "$3" ; Try to delete $InstDir
		RMDir "$3\.." ; (Optional) Try to delete the parent of $InstDir
	Pop $3
	Pop $2
	Exch $1 ; exitcode
FunctionEnd

;--------------------------------
;After Initialization Function
Function .onInit
	ReadRegStr $0 HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT}" "UninstallString"
	${If} $0 != ""
	${AndIf} ${Cmd} `MessageBox MB_YESNO|MB_ICONQUESTION "An installation of ${PRODUCT} exists in this directory ($INSTDIR). Do you want to uninstall this version?" /SD IDYES IDYES`
		!insertmacro UninstallExisting $0 $0
		${If} $0 <> 0
			MessageBox MB_YESNO|MB_ICONSTOP "Failed to uninstall, continue anyway?" /SD IDYES IDYES +2
			Abort
		${EndIf}
	${EndIf}
FunctionEnd

Function create_desktop_shortcut
	CreateShortcut "$DESKTOP\${PRODUCT}.lnk" "$INSTDIR\bin\sscanss.exe"
FunctionEnd
