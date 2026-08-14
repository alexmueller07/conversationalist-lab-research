; Conversation Analyst -- Windows installer.
;
; Per-user, no admin prompt: everything lands in %LOCALAPPDATA%, shortcuts
; in the user's Start Menu and Desktop, and an Add/Remove Programs entry so
; it uninstalls like anything else. The heavy scientific libraries are NOT
; inside this installer -- they install on first launch, visibly, which is
; what keeps this download small and the install instant.

!include "MUI2.nsh"

!define APPNAME "Conversation Analyst"
!define PUBLISHER "Alexander Mueller"
!define WEBSITE "https://conversation-analyst.vercel.app"
!ifndef VERSION
  !define VERSION "1.0.0"
!endif

Name "${APPNAME}"
OutFile "${OUTFILE}"
Unicode True
RequestExecutionLevel user
InstallDir "$LOCALAPPDATA\${APPNAME}"
SetCompressor /SOLID lzma

!define MUI_ICON "app.ico"
!define MUI_UNICON "app.ico"
!define MUI_ABORTWARNING

!define MUI_WELCOMEPAGE_TITLE "Conversation Analyst"
!define MUI_WELCOMEPAGE_TEXT "Measure a conversation the way the literature says to.$\r$\n$\r$\nThis installs the app for your user account only -- no administrator password needed. On first launch it downloads its analysis engine (about 1.6 GB), once.$\r$\n$\r$\nRecordings you analyze never leave your computer."
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_RUN
!define MUI_FINISHPAGE_RUN_FUNCTION LaunchApp
!define MUI_FINISHPAGE_RUN_TEXT "Start Conversation Analyst (runs first-time setup)"
!insertmacro MUI_PAGE_FINISH

Function LaunchApp
  ExecShell "open" "$INSTDIR\launch.vbs"
FunctionEnd

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

Section "Install"
  SetOutPath "$INSTDIR"
  File /r "payload\*.*"
  File "app.ico"

  CreateShortcut "$SMPROGRAMS\${APPNAME}.lnk" \
      "$WINDIR\System32\wscript.exe" '"$INSTDIR\launch.vbs"' \
      "$INSTDIR\app.ico" 0
  CreateShortcut "$DESKTOP\${APPNAME}.lnk" \
      "$WINDIR\System32\wscript.exe" '"$INSTDIR\launch.vbs"' \
      "$INSTDIR\app.ico" 0

  WriteUninstaller "$INSTDIR\uninstall.exe"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" \
      "DisplayName" "${APPNAME}"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" \
      "DisplayIcon" "$INSTDIR\app.ico"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" \
      "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" \
      "Publisher" "${PUBLISHER}"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" \
      "URLInfoAbout" "${WEBSITE}"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" \
      "UninstallString" '"$INSTDIR\uninstall.exe"'
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" \
      "NoModify" 1
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" \
      "NoRepair" 1
SectionEnd

Section "Uninstall"
  ; Close a running copy first, or its locked files survive the removal.
  ; Filtered by path so only THIS app's processes are touched -- a blanket
  ; taskkill on pythonw would take down unrelated Python programs.
  ExecWait 'powershell -NoProfile -Command "Get-Process pythonw,python -ErrorAction SilentlyContinue | Where-Object { $$_.Path -like \"*$INSTDIR*\" } | Stop-Process -Force"'

  ; If setup installed a private Python runtime, remove it through its own
  ; uninstaller so nothing stays registered; a runtime built as a venv from
  ; the user's Python is just a folder and needs no such step.
  IfFileExists "$INSTDIR\runtime-setup.exe" 0 +2
    ExecWait '"$INSTDIR\runtime-setup.exe" /uninstall /quiet'

  Delete "$SMPROGRAMS\${APPNAME}.lnk"
  Delete "$DESKTOP\${APPNAME}.lnk"
  RMDir /r "$INSTDIR"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}"
  ; The model cache (~27 MB in %USERPROFILE%\.convlab) and any results
  ; folders are the user's data and are deliberately left in place.
SectionEnd
