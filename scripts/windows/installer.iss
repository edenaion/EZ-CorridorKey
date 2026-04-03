; EZ-CorridorKey Windows Installer — Inno Setup 6
; Dark theme matching EZSCAPE brand identity
;
; Prerequisites:
;   1. Build frozen exe: pyinstaller installers/corridorkey-windows.spec --noconfirm
;   2. Compile: "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" scripts\windows\installer.iss
;
; Output: dist\EZ-CorridorKey-Setup-{version}.exe

#define MyAppName "EZ-CorridorKey"
#define MyAppVersion "1.9.0"
#define MyAppPublisher "EZscape"
#define MyAppURL "https://ezscape.space"
#define MyAppExeName "CorridorKey.exe"

[Setup]
AppId={{E7A3F1B2-9C4D-4E5F-8A6B-1C2D3E4F5A6B}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} v{#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
OutputDir=..\..\dist
OutputBaseFilename=EZ-CorridorKey-Setup-{#MyAppVersion}
SetupIconFile=..\..\ui\theme\corridorkey.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
WizardImageFile=wizard-large.bmp
WizardSmallImageFile=wizard-small.bmp
MinVersion=10.0
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
DisableProgramGroupPage=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\..\dist\CorridorKey\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Messages]
WelcomeLabel2=This will install [name/ver] on your computer.%n%nAI-powered green screen keying for VFX artists.%n%nRequires an NVIDIA GPU (GTX 10-series or newer).
FinishedLabel={#MyAppName} has been installed.%n%nOn first launch, the setup wizard will download the AI model (~383 MB).

[Code]
const
  BG_DARK      = $0E0A0A;
  BG_PANEL     = $141010;
  ACCENT_GOLD  = $03F2FF;   // CorridorKey brand yellow (#FFF203) in BGR
  NEON_GREEN   = $64E600;
  TEXT_LIGHT   = $D2C8C8;
  TEXT_DIM     = $6E6464;

procedure LinkClick(Sender: TObject);
var
  ErrorCode: Integer;
begin
  if TNewStaticText(Sender).Tag = 1 then
    ShellExec('open', 'https://discord.gg/2fgZNKyNza', '', '', SW_SHOW, ewNoWait, ErrorCode)
  else
    ShellExec('open', 'https://ezscape.space', '', '', SW_SHOW, ewNoWait, ErrorCode);
end;

procedure InitializeWizard();
var
  I: Integer;
begin
  // Main form
  WizardForm.Color := BG_DARK;

  // Top panel (header)
  WizardForm.MainPanel.Color := BG_PANEL;
  WizardForm.PageNameLabel.Font.Color := ACCENT_GOLD;
  WizardForm.PageNameLabel.Font.Style := [fsBold];
  WizardForm.PageDescriptionLabel.Font.Color := TEXT_DIM;

  // All inner pages — dark background
  WizardForm.InnerPage.Color := BG_DARK;
  WizardForm.WelcomePage.Color := BG_DARK;
  WizardForm.FinishedPage.Color := BG_DARK;
  WizardForm.InstallingPage.Color := BG_DARK;
  WizardForm.SelectDirPage.Color := BG_DARK;
  WizardForm.SelectTasksPage.Color := BG_DARK;
  WizardForm.ReadyPage.Color := BG_DARK;
  WizardForm.SelectProgramGroupPage.Color := BG_DARK;

  // Welcome page
  WizardForm.WelcomeLabel1.Font.Color := ACCENT_GOLD;
  WizardForm.WelcomeLabel1.Font.Size := 14;
  WizardForm.WelcomeLabel2.Font.Color := TEXT_LIGHT;

  // Finished page
  WizardForm.FinishedHeadingLabel.Font.Color := ACCENT_GOLD;
  WizardForm.FinishedHeadingLabel.Font.Size := 14;
  WizardForm.FinishedLabel.Font.Color := TEXT_LIGHT;
  WizardForm.YesRadio.Font.Color := TEXT_LIGHT;
  WizardForm.NoRadio.Font.Color := TEXT_LIGHT;
  WizardForm.RunList.Color := BG_DARK;
  WizardForm.RunList.Font.Color := TEXT_LIGHT;

  // Installing page
  WizardForm.StatusLabel.Font.Color := NEON_GREEN;
  WizardForm.FilenameLabel.Font.Color := TEXT_DIM;

  // Ready page
  WizardForm.ReadyLabel.Font.Color := NEON_GREEN;
  WizardForm.ReadyMemo.Color := BG_PANEL;
  WizardForm.ReadyMemo.Font.Color := TEXT_LIGHT;

  // Select dir page — ALL labels must be light
  WizardForm.SelectDirLabel.Font.Color := TEXT_LIGHT;
  WizardForm.SelectDirBrowseLabel.Font.Color := TEXT_LIGHT;
  WizardForm.DirEdit.Color := BG_PANEL;
  WizardForm.DirEdit.Font.Color := TEXT_LIGHT;

  // Disk space label
  WizardForm.DiskSpaceLabel.Font.Color := TEXT_DIM;

  // Tasks page
  WizardForm.SelectTasksLabel.Font.Color := TEXT_LIGHT;
  WizardForm.TasksList.Color := BG_PANEL;
  WizardForm.TasksList.Font.Color := TEXT_LIGHT;

  // Set font color for ALL child labels on every page to catch stragglers
  for I := 0 to WizardForm.ComponentCount - 1 do
  begin
    if WizardForm.Components[I] is TNewStaticText then
      TNewStaticText(WizardForm.Components[I]).Font.Color := TEXT_LIGHT;
    if WizardForm.Components[I] is TLabel then
      TLabel(WizardForm.Components[I]).Font.Color := TEXT_LIGHT;
    if WizardForm.Components[I] is TNewCheckListBox then
    begin
      TNewCheckListBox(WizardForm.Components[I]).Color := BG_DARK;
      TNewCheckListBox(WizardForm.Components[I]).Font.Color := TEXT_LIGHT;
    end;
  end;

  // Re-apply accent colors after the bulk sweep
  WizardForm.PageNameLabel.Font.Color := ACCENT_GOLD;
  WizardForm.PageDescriptionLabel.Font.Color := TEXT_DIM;
  WizardForm.WelcomeLabel1.Font.Color := ACCENT_GOLD;
  WizardForm.FinishedHeadingLabel.Font.Color := ACCENT_GOLD;
  WizardForm.StatusLabel.Font.Color := NEON_GREEN;
  WizardForm.ReadyLabel.Font.Color := NEON_GREEN;
  WizardForm.DiskSpaceLabel.Font.Color := TEXT_DIM;
  WizardForm.FilenameLabel.Font.Color := TEXT_DIM;

  // Hide bevels for clean dark look
  WizardForm.Bevel.Visible := False;
  WizardForm.Bevel1.Visible := False;

  // Clickable Discord link on finished page
  with TNewStaticText.Create(WizardForm.FinishedPage) do
  begin
    Parent := WizardForm.FinishedPage;
    Left := WizardForm.FinishedLabel.Left;
    Top := WizardForm.FinishedLabel.Top + WizardForm.FinishedLabel.Height + 16;
    Caption := 'Join EZSCAPE Discord';
    Font.Color := NEON_GREEN;
    Font.Style := [fsUnderline];
    Font.Size := 9;
    Cursor := crHand;
    Tag := 1;  // Discord
    OnClick := @LinkClick;
  end;

  // Copyright + website link
  with TNewStaticText.Create(WizardForm.FinishedPage) do
  begin
    Parent := WizardForm.FinishedPage;
    Left := WizardForm.FinishedLabel.Left;
    Top := WizardForm.FinishedLabel.Top + WizardForm.FinishedLabel.Height + 42;
    Caption := #169 + ' EZscape  |  ezscape.space';
    Font.Color := TEXT_DIM;
    Font.Size := 8;
    Cursor := crHand;
    Tag := 2;  // Website
    OnClick := @LinkClick;
  end;
end;

// Uninstall: optionally clean up user data
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  AppDataDir: String;
  ConfigDir: String;
  Msg: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    AppDataDir := ExpandConstant('{userappdata}\EZ-CorridorKey');
    ConfigDir := ExpandConstant('{userappdata}\CorridorKey');

    if DirExists(AppDataDir) or DirExists(ConfigDir) then
    begin
      Msg := 'Do you also want to remove your EZ-CorridorKey data?' + #13#10 +
             '(AI models, projects, and preferences)' + #13#10 + #13#10 +
             'Location: ' + AppDataDir + #13#10 + #13#10 +
             'Choose No to keep your data for future reinstalls.';
      if MsgBox(Msg, mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
      begin
        DelTree(AppDataDir, True, True, True);
        DelTree(ConfigDir, True, True, True);
        RegDeleteKeyIncludingSubkeys(HKCU, 'Software\EZSCAPE\EZ-CorridorKey');
        RegDeleteKeyIncludingSubkeys(HKCU, 'Software\EZSCAPE');
      end;
    end;
  end;
end;
