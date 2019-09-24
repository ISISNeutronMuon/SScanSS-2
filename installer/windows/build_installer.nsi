;--------------------------------
;Include Modern UI

  !include "MUI2.nsh"
  !include "version.nsh"

;--------------------------------
;General
  ;Properly display all languages
  Unicode true

  ;Define name of the product
  !define PRODUCT "SScanSS-2"
  
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
;Pages

  ;For the installer
  
  !define MUI_ICON "install.ico"
  !define MUI_UNICON "uninstall.ico"
  !define MUI_WELCOMEFINISHPAGE_BITMAP "banner.bmp"
  !define MUI_UNWELCOMEFINISHPAGE_BITMAP "banner.bmp"
  !define MUI_HEADERIMAGE
  !define MUI_HEADERIMAGE_RIGHT
  !define MUI_HEADERIMAGE_BITMAP "header.bmp"
  !define MUI_HEADERIMAGE_UNBITMAP "header.bmp"
  
  !insertmacro MUI_PAGE_WELCOME # simply remove this and other pages if you don't want it
  !insertmacro MUI_PAGE_LICENSE "..\bundle\LICENSE" # link to an ANSI encoded license file
  !insertmacro MUI_PAGE_INSTFILES
  
  !define MUI_FINISHPAGE_RUN	$INSTDIR\bin\sscanss.exe
  !define MUI_FINISHPAGE_RUN_NOTCHECKED
  ;!define MUI_FINISHPAGE_SHOWREADME_NOTCHECKED
  ;!define MUI_FINISHPAGE_SHOWREADME $INSTDIR\readme.txt
  ;!define MUI_FINISHPAGE_LINK link_text
  
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

Section "Main Component"
  SectionIn RO # Just means if in component mode this is locked

  ;Set output path to the installation directory.
  SetOutPath $INSTDIR

  ;Put the following file in the SetOutPath
  File /r "..\bundle\bin"
  File /r "..\bundle\instruments"
  File /r "..\bundle\static"
  File "..\bundle\LICENSE"
  File "..\bundle\logging.json"

  ;Store installation folder in registry
  WriteRegStr HKLM "Software\${PRODUCT}" "" $INSTDIR

  ;Registry information for add/remove programs
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT}" "DisplayName" "${PRODUCT}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT}" "UninstallString" '"$INSTDIR\uninstall.exe"'
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT}" "NoModify" 1
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT}" "NoRepair" 1

  ;Create optional start menu shortcut for uninstaller and Main component
  CreateShortCut "$SMPROGRAMS\${PRODUCT}.lnk" "$INSTDIR\bin\sscanss.exe" "" "$INSTDIR\bin\sscanss.exe" 0
  
  ;Create uninstaller
  WriteUninstaller "uninstall.exe"

SectionEnd

;--------------------------------
;Uninstaller Section

Section "Uninstall"

  ;Remove all registry keys
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT}"
  DeleteRegKey HKLM "Software\${PRODUCT}"

  ;Delete the installation directory + all files in it
  ;Add 'RMDir /r "$INSTDIR\folder\*.*"' for every folder you have added additionaly
  RMDir /r "$INSTDIR\*.*"
  RMDir "$INSTDIR"
  
  ;Delete Start Menu Shortcuts
  Delete "$SMPROGRAMS\${PRODUCT}.lnk"

SectionEnd


;--------------------------------
;After Initialization Function

 Function .onInit
   ${If} ${FileExists} $INSTDIR\bin\sscanss.exe
		MessageBox MB_OK "A previous installation of SScanSS-2 exist in this directory ($INSTDIR). Please uninstall the previous version before proceeding with a new installation."
		Abort
   ${EndIf}
 FunctionEnd
