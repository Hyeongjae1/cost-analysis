<#
.SYNOPSIS
  과거 커밋을 기준으로 새 브랜치를 만들어 안전하게 내용을 확인합니다. (develop을 덮어쓰지 않음)
.PARAMETER List
  최근 커밋 25개만 표시하고 종료합니다.
.PARAMETER Commit
  복구 기준 커밋 해시(전체 또는 앞 7자 이상).
.EXAMPLE
  .\scripts\git-restore.ps1 -List
  .\scripts\git-restore.ps1 -Commit 0c8d9ec
#>
param(
    [switch] $List,
    [string] $Commit = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$git = "git"
try { $null = Get-Command git -ErrorAction Stop } catch {
    $git = "C:\Program Files\Git\bin\git.exe"
}

if ($List) {
    & $git log --oneline -25
    exit 0
}

if (-not $Commit.Trim()) {
    & $git log --oneline -15
    $Commit = Read-Host "복구 기준 커밋 해시(위 목록에서 복사)"
}
if (-not $Commit.Trim()) {
    Write-Host "커밋이 지정되지 않았습니다." -ForegroundColor Yellow
    exit 1
}

$branchName = "restore-{0}" -f (Get-Date -Format "yyyyMMdd-HHmmss")
& $git switch -c $branchName $Commit.Trim()
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "브랜치 '$branchName' 에서 과거 커밋 $Commit 상태를 열었습니다." -ForegroundColor Green
Write-Host "develop 으로 돌아가려면:  git switch develop" -ForegroundColor Cyan
Write-Host "이 브랜치 내용을 develop 에 합치려면(주의):  git switch develop  후  git merge $branchName" -ForegroundColor Cyan
Write-Host "develop 을 과거 커밋으로 강제 맞추려면(위험, 협업 시 비권장):  git switch develop  후  git reset --hard $Commit  그리고  git push --force-with-lease" -ForegroundColor DarkYellow
