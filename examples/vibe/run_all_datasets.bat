@echo off
setlocal enabledelayedexpansion

pushd "%~dp0"

set "DATASETS=agnews-mxbai arxiv-nomic gooaq-distilroberta inaturalist-resnet laion-clip landmark-dino yahoo-minilm yandex imagenet-align"
set "FAILED="

for %%D in (%DATASETS%) do (
    echo.
    echo ================================================================================
    echo Running VIBE benchmark for dataset: %%D
    echo ================================================================================
    uv run python main.py --dataset %%D --algorithm deg-qg --no-show %*
    if errorlevel 1 (
        echo [ERROR] Dataset %%D failed with exit code !errorlevel!.
        set "FAILED=!FAILED! %%D"
    )
)

echo.
echo ================================================================================
if defined FAILED (
    echo Finished with failures:%FAILED%
    popd
    exit /b 1
) else (
    echo All VIBE datasets completed successfully.
    popd
    exit /b 0
)
