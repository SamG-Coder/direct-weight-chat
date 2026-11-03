Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
python -m chat1gb_autoregressive.chat @args
