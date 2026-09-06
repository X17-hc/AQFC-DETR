param(
  [Parameter(Mandatory=$true)][string]$DataRoot,
  [string]$OutputDir = "outputs\aitodv2_main"
)
python main.py --config configs/aitodv2/aqfc_r50_5scale_24e.py `
  --data-root $DataRoot --output-dir $OutputDir
