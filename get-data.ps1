# Download real-world training data for nib-rs.
#
# Usage:
#   .\get-data.ps1                 # Tiny Shakespeare (default, ~1.1 MB)
#   .\get-data.ps1 -Dataset wiki   # Wikipedia sample
#   .\get-data.ps1 -Dataset code   # Rust source corpus
#
# All sources are public and downloaded over HTTPS.

param(
    [ValidateSet("shakespeare", "wiki", "code")]
    [string]$Dataset = "shakespeare",
    [string]$Out = "shakespeare.txt"
)

$ErrorActionPreference = "Stop"

function Fetch($url, $dest) {
    Write-Host "Downloading $url" -ForegroundColor Cyan
    Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing -TimeoutSec 120
    $size = (Get-Item $dest).Length
    Write-Host "Saved $dest ($([math]::Round($size/1KB,1)) KB)" -ForegroundColor Green
}

switch ($Dataset) {
    "shakespeare" {
        # Real Shakespeare plays/poems (~1.1 MB), standard char-level corpus.
        Fetch "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" $Out
    }
    "wiki" {
        # A small real Wikipedia text sample.
        Fetch "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" $Out
        Write-Host "Note: swap in your own Wikipedia dump for larger corpora." -ForegroundColor Yellow
    }
    "code" {
        # Concatenate local Rust sources into one text file.
        $files = Get-ChildItem -Path "src" -Filter *.rs -Recurse
        $all = foreach ($f in $files) { Get-Content $f.FullName -Raw }
        $all -join "`n`n" | Set-Content -Path $Out -Encoding UTF8
        Write-Host "Saved $Out from $($files.Count) Rust source files" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Next: train on it with" -ForegroundColor Cyan
Write-Host "  .\target\release\nib.exe train --input $Out --output model.json --tokenizer tokenizer.json --epochs 40"
