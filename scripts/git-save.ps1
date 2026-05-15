<#
.SYNOPSIS
  스테이징 후 커밋·푸시. 메시지 형식: [NNN] yyyy-MM-dd HH:mm:ss <이름>
  -Message 생략 시 대화형으로 이름(커밋 메시지)을 묻습니다. 비우면 종료합니다.
.EXAMPLE
  .\scripts\git-save.ps1 -Message "대시보드 레이아웃 조정"
  .\scripts\git-save.ps1
#>
param(
    [Parameter(Mandatory = $false)]
    [string] $Message = "",
    [string] $Remote = "origin",
    [string] $Branch = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$git = "git"
try { $null = Get-Command git -ErrorAction Stop } catch {
    $git = "C:\Program Files\Git\bin\git.exe"
}

if (-not $Branch.Trim()) {
    $Branch = (& $git rev-parse --abbrev-ref HEAD).Trim()
}

if (-not $Message.Trim()) {
    $Message = Read-Host "저장할 이름(커밋 메시지)을 입력하세요"
}
if (-not $Message.Trim()) {
    Write-Host "메시지가 비어 있어 커밋하지 않습니다." -ForegroundColor Yellow
    exit 1
}

$seqPath = Join-Path $PSScriptRoot "git-commit-seq.txt"
$seq = 1
if (Test-Path $seqPath) {
    $raw = (Get-Content $seqPath -Raw).Trim()
    $parsed = 0
    if ([int]::TryParse($raw, [ref]$parsed)) { $seq = $parsed }
}
$ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$fullMessage = ("[{0:D3}] {1} {2}" -f $seq, $ts, $Message.Trim())

& $git add -A
$st = & $git status --porcelain
if (-not $st) {
    Write-Host "커밋할 변경이 없습니다. 작업 내용을 저장한 뒤 다시 실행하세요." -ForegroundColor Yellow
    exit 0
}

& $git commit -m $fullMessage
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$next = $seq + 1
Set-Content -Path $seqPath -Value "$next" -NoNewline

& $git push $Remote $Branch
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "완료: $fullMessage  ($Remote $Branch)" -ForegroundColor Green
