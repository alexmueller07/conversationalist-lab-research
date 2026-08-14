' Conversation Analyst launcher.
'
' The shortcut points here. If setup has not run yet, the visible setup
' console runs once; afterwards the app starts silently with no console
' window flashing.

Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
base = fso.GetParentFolderName(WScript.ScriptFullName)

flag = base & "\installed.flag"

If Not fso.FileExists(flag) Then
    shell.Run """" & base & "\setup.bat""", 1, False
Else
    Set f = fso.OpenTextFile(flag, 1)
    pythonw = Trim(f.ReadLine)
    f.Close
    If Not fso.FileExists(pythonw) Then
        ' The environment was moved or deleted; run setup again.
        shell.Run """" & base & "\setup.bat""", 1, False
    Else
        shell.CurrentDirectory = base & "\app"
        shell.Run """" & pythonw & """ -m conversation_analyst.gui", 0, False
    End If
End If
