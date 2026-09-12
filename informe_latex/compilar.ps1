$ErrorActionPreference = "Stop"
$directorio = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $directorio
try {
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    bibtex main
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    Copy-Item main.pdf Informe_Error_Numerico.pdf -Force
}
finally {
    Pop-Location
}
