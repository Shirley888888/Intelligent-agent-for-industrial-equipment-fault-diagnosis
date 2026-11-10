@echo off
setlocal
cd /d "%~dp0"

REM Configure the project-specific LLM planner factory before enabling unattended research.
REM The factory must return hypothesis_builder(papers, memory) -> {statement, rationale, plan}.
set ATREA_REPO_ROOT=%CD%
set ATREA_EXPERIMENT_RUNNER=atrea.chronos2_runner:build_runner_from_env
set ATREA_CHRONOS2_CONFIG=%CD%\configs\chronos2_daily.example.json
set ATREA_WORKSPACE=%CD%\agent_workspace\atrea_experience.sqlite3
set ATREA_PAPER_QUERY=industrial transformer oil temperature time series forecasting foundation model

if "%ATREA_HYPOTHESIS_BUILDER%"=="" (
  echo ERROR: Set ATREA_HYPOTHESIS_BUILDER=your_module:build_hypothesis_builder before running unattended research.
  exit /b 2
)

python -m atrea.scheduler --callback atrea.daily_job_template:run_daily_job --state agent_workspace\daily_scheduler.json --hour 8 --minute 0
