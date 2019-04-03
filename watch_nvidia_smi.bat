@ECHO OFF
:loop
  cls
  nvidia-smi.exe
  timeout /t 1 > NUL
goto loop