@echo off 
SETLOCAL EnableDelayedExpansion
echo *********************************************
echo ***************STDN_NAS_keras****************
echo Phase Select:
echo 1. Run All ( train, search, retrain, test )
echo 2. Train
echo 3. Search
echo 4. Retrain
echo 5. Test
echo 6. Quit
echo *********************************************

@REM change the conda environment here
@REM vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
SET conda-env=stdn_keras
call activate %conda-env%

CHOICE /C:123456 /M Select

GOTO :case-%ERRORLEVEL%


:case-1
    echo starting training phase...
    powershell "cmd /c python train_supernet.py | tee log/training_step.txt"
    echo starting searching phase...
    powershell "cmd /c python search.py"
    echo starting retaining phase...
    powershell "cmd /c python retrain_architecture.py | tee log/retraining_step.txt"
    echo starting testing phase...
    powershell "cmd /c python eval_architecture.py"
    echo all phase complete
    GOTO :exit_prog

:case-2
    echo starting training phase...
    powershell "cmd /c python train_supernet.py | tee log/training_step.txt"
    @REM python train_supernet.py > log/taining_supernet.txt 2>&1 & type log/taining_supernet.txt
    echo training phase complete
    GOTO :exit_prog

:case-3 
    echo starting searching phase...
    powershell "cmd /c python search.py"
    echo searching phase complete
    GOTO :exit_prog

:case-4
    echo starting retraining phase...
    powershell "cmd /c python retrain_architecture.py | tee log/retraining_step.txt"
    echo retaining phase complete
    GOTO :exit_prog

:case-5
    echo starting testing phase...
    powershell "cmd /c python eval_architecture.py"
    echo testing phase complete
    GOTO :exit_prog

:exit_prog
pause