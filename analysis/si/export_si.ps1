# Update Word fields (caption numbers, lists of figures/tables) in SI.docx, save it, and export SI.pdf.
# Usage (from the project root): powershell -NoProfile -File analysis\si\export_si.ps1
$ErrorActionPreference = 'Stop'
$docx = (Resolve-Path (Join-Path $PSScriptRoot 'SI.docx')).Path
$pdf = [IO.Path]::ChangeExtension($docx, '.pdf')
$word = New-Object -ComObject Word.Application
$preOpen = $word.Documents.Count
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($docx, $false, $false, $false)
    $doc.Fields.Update() | Out-Null                      # SEQ caption numbers
    for ($i = 1; $i -le $doc.TablesOfFigures.Count; $i++) { $doc.TablesOfFigures.Item($i).Update() }
    $doc.Repaginate()
    for ($i = 1; $i -le $doc.TablesOfFigures.Count; $i++) { $doc.TablesOfFigures.Item($i).UpdatePageNumbers() }
    $doc.Save()
    $doc.ExportAsFixedFormat($pdf, 17, $false, 0, 0, 0, 0, 0, $true, $true, 0, $true, $true, $false)
    "pages: $($doc.ComputeStatistics(2))"
    for ($i = 1; $i -le $doc.TablesOfFigures.Count; $i++) {
        "--- list $i"
        $doc.TablesOfFigures.Item($i).Range.Paragraphs | ForEach-Object { $_.Range.Text.Trim() } | Where-Object { $_ }
    }
    $doc.Close([ref]0)
} finally {
    if ($word.Documents.Count -eq 0 -and $preOpen -eq 0) { $word.Quit() }
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word)
}
