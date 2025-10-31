; Inno Setup Configuration for Signal Analyzer
; Download Inno Setup from: https://jrsoftware.org/isdl.php
; To build: Right-click this file and select "Compile"

[Setup]
; Basic Information
AppName=Signal Analyzer
AppVersion=1.0.2
AppPublisher=Signal Analysis Lab
AppPublisherURL=https://github.com/yourusername/DataChaEnhanced
AppSupportURL=https://github.com/yourusername/DataChaEnhanced/issues
AppUpdatesURL=https://github.com/yourusername/DataChaEnhanced/releases
DefaultDirName={autopf}\SignalAnalyzer
DefaultGroupName=Signal Analyzer
AllowNoIcons=yes
OutputDir=installers
OutputBaseFilename=SignalAnalyzer_Setup_v1.0.2
SetupIconFile=assets\icon.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern

; Privileges
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog

; Architecture
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

; Version Information
VersionInfoVersion=1.0.2
VersionInfoCompany=Signal Analysis Lab
VersionInfoDescription=Signal Analyzer Installer
VersionInfoCopyright=Copyright (C) 2025

; Uninstall
UninstallDisplayName=Signal Analyzer
UninstallDisplayIcon={app}\SignalAnalyzer.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 6.1; Check: not IsAdminInstallMode

[Files]
; Main application files
Source: "dist\SignalAnalyzer\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Documentation
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion; DestName: "README.txt"
Source: "User_Guide.txt"; DestDir: "{app}"; Flags: ignoreversion; AfterInstall: ConvertLineEndings

[Icons]
; Start Menu
Name: "{group}\Signal Analyzer"; Filename: "{app}\SignalAnalyzer.exe"; WorkingDir: "{app}"; Comment: "Advanced Signal Processing Tool"
Name: "{group}\User Guide"; Filename: "{app}\User_Guide.txt"
Name: "{group}\{cm:UninstallProgram,Signal Analyzer}"; Filename: "{uninstallexe}"

; Desktop icon (optional)
Name: "{autodesktop}\Signal Analyzer"; Filename: "{app}\SignalAnalyzer.exe"; WorkingDir: "{app}"; Tasks: desktopicon

; Quick Launch (optional, for older Windows)
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\Signal Analyzer"; Filename: "{app}\SignalAnalyzer.exe"; Tasks: quicklaunchicon

[Registry]
; Register .atf file association
Root: HKCR; Subkey: ".atf"; ValueType: string; ValueName: ""; ValueData: "SignalAnalyzer.DataFile"; Flags: uninsdeletevalue
Root: HKCR; Subkey: "SignalAnalyzer.DataFile"; ValueType: string; ValueName: ""; ValueData: "Signal Analyzer Data File"; Flags: uninsdeletekey
Root: HKCR; Subkey: "SignalAnalyzer.DataFile\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\SignalAnalyzer.exe,0"
Root: HKCR; Subkey: "SignalAnalyzer.DataFile\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\SignalAnalyzer.exe"" ""%1"""

[Run]
; Option to launch after installation
Filename: "{app}\SignalAnalyzer.exe"; Description: "{cm:LaunchProgram,Signal Analyzer}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Clean up any user-generated files if user wants to
Type: filesandordirs; Name: "{app}\logs"
Type: filesandordirs; Name: "{app}\data"

[Code]
procedure ConvertLineEndings;
begin
  // Additional setup code can go here
end;

function InitializeSetup(): Boolean;
var
  ResultCode: Integer;
begin
  Result := True;
  
  // Check if .NET Framework or other prerequisites are installed
  // Add custom checks here if needed
  
  // Example: Check for minimum Windows version
  if not (GetWindowsVersion shr 24 >= 10) then
  begin
    if MsgBox('Signal Analyzer works best on Windows 10 or later. ' +
              'You appear to be running an older version. ' +
              'Do you want to continue anyway?', 
              mbConfirmation, MB_YESNO) = IDNO then
    begin
      Result := False;
    end;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    // Post-installation tasks
    // For example, create initial configuration files
  end;
end;

function UninstallNeedRestart(): Boolean;
begin
  Result := False;
end;

