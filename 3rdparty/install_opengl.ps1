# Mesa DLLs found linked from:
#     http://qt-project.org/wiki/Cross-compiling-Mesa-for-Windows
# to:
#     http://sourceforge.net/projects/msys2/files/REPOS/MINGW/x86_64/mingw-w64-x86_64-mesa-10.2.4-1-any.pkg.tar.xz/download

function InstallMesaOpenGL () {

    $url = "3rdparty\opengl32_mingw_64.dll"
    $filepath = "C:\Windows\system32\opengl32.dll"
    
    takeown /F $filepath /A
    icacls $filepath /grant "${env:ComputerName}\${env:UserName}:F"
    Remove-item -LiteralPath $filepath
    Write-Host "Copying" $url

	Copy-Item $url -Destination $filepath
	
    if (Test-Path $filepath) {
        Write-Host "File saved at" $filepath
    } 
}


function main () {
    InstallMesaOpenGL
}

main