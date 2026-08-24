<#
push_to_github.ps1
一键将当前文件夹推送到指定 GitHub 仓库（交互式凭据提示，推荐在本地运行）

说明：
- 在项目根目录放置并从该目录运行： .\push_to_github.ps1
- 脚本不会把你的 PAT 写入任何文件；推送时会弹出凭据对话或在命令行提示输入用户名与密码（将 PAT 当作密码粘贴）。
- 如果未设置 git user.name 或 user.email，会提示输入或使用临时占位符（建议填写真实信息）。
- 如果 PowerShell 阻止执行脚本，可在当前会话临时允许： Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force
#>

Param(
    [string]$RepoPath = (Get-Location).Path,
    [string]$RemoteUrl = "https://github.com/Shirley888888/Intelligent-agent-for-industrial-equipment-fault-diagnosis.git"
)

Write-Host "Repository path:" -ForegroundColor Cyan $RepoPath
Push-Location $RepoPath

# Helper to run git and capture output
function Run-Git {
    $args = $args
    & git @args
}

# Check git availability
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "Git not found in PATH. Please install Git for Windows: https://git-scm.com/download/win"
    Pop-Location
    exit 1
}

Write-Host "Git found:" (git --version) -ForegroundColor Green

# Ensure user.name and user.email are set for commits
$uname = git config user.name
$uemail = git config user.email
if (-not $uname -or -not $uemail) {
    Write-Host "git user.name or user.email not set for this environment." -ForegroundColor Yellow
    $setName = Read-Host "Enter git user.name to use for commits (leave blank to set 'GitUser')"
    if ([string]::IsNullOrWhiteSpace($setName)) { $setName = "GitUser" }
    git config user.name "$setName"

    $setEmail = Read-Host "Enter git user.email to use for commits (leave blank to set 'gituser@example.com')"
    if ([string]::IsNullOrWhiteSpace($setEmail)) { $setEmail = "gituser@example.com" }
    git config user.email "$setEmail"
    Write-Host "Set local git user.name and user.email for this repo." -ForegroundColor Green
}

# Initialize repo if needed
if (-not (Test-Path ".git")) {
    Write-Host "Initializing new git repository..." -ForegroundColor Cyan
    git init
} else {
    Write-Host "Existing git repository found." -ForegroundColor Cyan
}

# Stage changes
Write-Host "Staging all files..." -ForegroundColor Cyan
git add .

# Commit if there are changes
$porcelain = git status --porcelain
if (-not [string]::IsNullOrWhiteSpace($porcelain)) {
    $commitMessage = Read-Host "Enter commit message (default: 'Update from push_to_github.ps1')"
    if ([string]::IsNullOrWhiteSpace($commitMessage)) { $commitMessage = "Update from push_to_github.ps1" }
    git commit -m "$commitMessage"
    Write-Host "Committed changes." -ForegroundColor Green
} else {
    Write-Host "No changes to commit." -ForegroundColor Yellow
}

# Ensure branch name is 'main'
Write-Host "Setting branch to 'main'..." -ForegroundColor Cyan
# If HEAD doesn't exist (no commits), create the branch after initial commit
try {
    git rev-parse --verify HEAD > $null 2>&1
    $hasHead = $true
} catch {
    $hasHead = $false
}

if (-not $hasHead) {
    # No commits yet: create an orphan commit if necessary
    Write-Host "No commits found in repo; creating an initial empty commit to establish branch 'main'..." -ForegroundColor Cyan
    git commit --allow-empty -m "Initial commit (created by push_to_github.ps1)"
}

git branch -M main

# Add or update remote 'origin'
$hasOrigin = (git remote | Select-String -Pattern "^origin$" -Quiet)
if ($hasOrigin) {
    Write-Host "Updating existing remote 'origin' to: $RemoteUrl" -ForegroundColor Cyan
    git remote set-url origin $RemoteUrl
} else {
    Write-Host "Adding remote 'origin': $RemoteUrl" -ForegroundColor Cyan
    git remote add origin $RemoteUrl
}

# Enable Git Credential Manager (manager-core) if not configured
$helper = git config --global credential.helper
if ([string]::IsNullOrWhiteSpace($helper)) {
    Write-Host "Enabling Git Credential Manager (manager-core) globally to ensure interactive credential prompt..." -ForegroundColor Cyan
    try {
        git config --global credential.helper manager-core
        Write-Host "Enabled credential.helper=manager-core" -ForegroundColor Green
    } catch {
        Write-Host "Failed to set global credential helper; continuing and relying on git to prompt for credentials." -ForegroundColor Yellow
    }
} else {
    Write-Host "Existing credential helper: $helper" -ForegroundColor Cyan
}

Write-Host "\nReady to push to origin/main." -ForegroundColor Magenta
Write-Host "When prompted, enter your GitHub username; paste your Personal Access Token (PAT) as the password." -ForegroundColor Magenta

$null = Read-Host "Press Enter to start git push (or Ctrl+C to cancel)"

# Perform push
try {
    git push -u origin main
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Push succeeded. Open the repository URL to verify: $RemoteUrl" -ForegroundColor Green
    } else {
        Write-Host "git push exited with code $LASTEXITCODE. If authentication failed, ensure you used your username and PAT as password. Consider running 'git remote -v' and troubleshooting." -ForegroundColor Red
    }
} catch {
    Write-Host "An error occurred during git push: $_" -ForegroundColor Red
}

Pop-Location

Write-Host "Done." -ForegroundColor Cyan
Write-Host "Security reminder: do not paste your PAT into files or share it. Revoke or limit the PAT after use if appropriate." -ForegroundColor Yellow
