[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
$base = "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/"

# file name, title, updated-as-on date, PDF name on RBI server
$docs = @(
 @("rbi_cb_credit_facilities.pdf","Commercial Banks - Credit Facilities","Oct 1, 2026","154MD.PDF"),
 @("rbi_cb_kyc.pdf","Commercial Banks - Know Your Customer","Oct 1, 2026","169MD.PDF"),
 @("rbi_cb_responsible_business_conduct.pdf","Commercial Banks - Responsible Business Conduct","Oct 1, 2026","170MD.PDF"),
 @("rbi_cb_credit_debit_cards.pdf","Commercial Banks - Credit Cards and Debit Cards","Oct 1, 2026","155MD.PDF"),
 @("rbi_cb_resolution_stressed_assets.pdf","Commercial Banks - Resolution of Stressed Assets","Oct 1, 2026","165MD.PDF"),
 @("rbi_cb_income_recognition_asset_classification.pdf","Commercial Banks - Income Recognition, Asset Classification and Provisioning","Oct 1, 2026","164MD.PDF"),
 @("rbi_cb_managing_risks_outsourcing.pdf","Commercial Banks - Managing Risks in Outsourcing","-","171MD.PDF"),
 @("rbi_cb_interest_rates_on_advances.pdf","Commercial Banks - Interest Rates on Advances","-","161MD.PDF"),
 @("rbi_nbfc_credit_facilities.pdf","NBFC - Credit Facilities","Jul 15, 2026","347MD5CC21D3597C04354B67A42A1A4CB439C.PDF"),
 @("rbi_nbfc_kyc.pdf","NBFC - Know Your Customer","Dec 29, 2025","361MD1E2F8EA063454AD5AFA1D02A1BA5ACA7.PDF"),
 @("rbi_nbfc_responsible_business_conduct.pdf","NBFC - Responsible Business Conduct","Jul 1, 2026","362MD26CA543937BA439A97E1BCFC08CF5808.PDF"),
 @("rbi_nbfc_credit_cards.pdf","NBFC - Credit Cards: Issuance and Conduct","-","348MD993245D8570047299D1814811F20AFEE.PDF"),
 @("rbi_nbfc_resolution_stressed_assets.pdf","NBFC - Resolution of Stressed Assets","Oct 1, 2026","357MD501BDDC9758E40B592D1AD2D919CC6AF.PDF"),
 @("rbi_nbfc_income_recognition_asset_classification.pdf","NBFC - Income Recognition, Asset Classification and Provisioning","Oct 1, 2026","356MD4F8109CA54BE44A9805C5300601F8A11.PDF"),
 @("rbi_nbfc_managing_risks_outsourcing.pdf","NBFC - Managing Risks in Outsourcing","-","363MD40F22B0CDF734E3C884A9CBCE5DF9194.PDF"),
 @("rbi_nbfc_transfer_distribution_credit_risk.pdf","NBFC - Transfer and Distribution of Credit Risk","Jul 1, 2026","352MDCF4E55B0ACD24FCC8AED1C715997F0F9.PDF"),
 @("rbi_nbfc_securitisation.pdf","NBFC - Securitisation Transactions","Jul 1, 2026","353MD9A35481FC07E4D1199A62E5C2356B0C4.PDF"),
 @("rbi_sfb_credit_facilities.pdf","Small Finance Banks - Credit Facilities","Oct 1, 2026","185MD.PDF"),
 @("rbi_sfb_kyc.pdf","Small Finance Banks - Know Your Customer","Oct 1, 2026","200MD.PDF"),
 @("rbi_sfb_responsible_business_conduct.pdf","Small Finance Banks - Responsible Business Conduct","Oct 1, 2026","201MD.PDF"),
 @("rbi_sfb_credit_debit_cards.pdf","Small Finance Banks - Credit Cards and Debit Cards","Oct 1, 2026","186MD.PDF")
)

New-Item -ItemType Directory -Force data\raw | Out-Null
"| file | title | issued | updated as on | web address |" | Set-Content data\SOURCES.md
"| --- | --- | --- | --- | --- |" | Add-Content data\SOURCES.md

foreach ($d in $docs) {
  $url = $base + $d[3]
  Write-Host "Downloading $($d[0]) ..."
  try {
    Invoke-WebRequest -Uri $url -OutFile ("data\raw\" + $d[0]) -UserAgent $ua -UseBasicParsing
    "| $($d[0]) | $($d[1]) | Nov 28, 2025 | $($d[2]) | $url |" | Add-Content data\SOURCES.md
  } catch {
    Write-Host "FAILED: $($d[0]) - $($_.Exception.Message)" -ForegroundColor Red
  }
  Start-Sleep -Seconds 1
}
Write-Host "Done."