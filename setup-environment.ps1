$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPath = Join-Path $projectRoot 'venv'
$pythonCommand = Get-Command py -ErrorAction SilentlyContinue

if ($null -ne $pythonCommand) {
    $pythonArgs = @('-3.14')
} else {
    $pythonCommand = Get-Command python -ErrorAction Stop
    $pythonArgs = @()
}

Write-Host 'Creating the virtual environment without ensurepip...'
& $pythonCommand.Source @pythonArgs -m venv --clear --without-pip $venvPath

Write-Host 'Installing pip into the virtual environment...'
& $pythonCommand.Source @pythonArgs -m pip --python $venvPath install --upgrade pip

$requirementsPath = Join-Path $projectRoot 'requirements.txt'
if (Test-Path $requirementsPath) {
    Write-Host 'Installing project dependencies...'
    & (Join-Path $venvPath 'Scripts\python.exe') -m pip install -r $requirementsPath
}

Write-Host "Environment ready: $venvPath"
Write-Host 'Activate it with: .\venv\Scripts\Activate.ps1'