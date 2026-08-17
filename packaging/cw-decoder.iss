; cw-decoder Windows インストーラ (Inno Setup 6)
;
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\cw-decoder.iss
;
; 先に PyInstaller のビルドを済ませておくこと (dist\cw-decoder\ を入力にする)。
; scripts\build_installer.py が両方まとめて行う。
;
; **管理者権限を要求しない。** インストール先をユーザ領域にしてある。
; 無線機の隣の PC が会社/家族共用で、管理者になれないことがあるため。

#define AppName      "cw-decoder"
#define AppVersion   "0.3.0"
#define AppPublisher "cw-decoder"
#define AppExeName   "cw-decoder.exe"
#define SourceDir    "..\dist\cw-decoder"

[Setup]
AppId={{7E2C4C41-9C2B-4A2E-9E0B-5B0D9E7A1C33}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
; ユーザ領域へ入れる = 管理者権限が要らない
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\dist
OutputBaseFilename=cw-decoder-{#AppVersion}-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; 64bit 版 Python で作っているので 64bit 専用
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#AppExeName}
; 日本語環境向け
ShowLanguageDialog=no

[Languages]
Name: "japanese"; MessagesFile: "compiler:Languages\Japanese.isl"

[Tasks]
Name: "desktopicon"; Description: "デスクトップにショートカットを作る"; GroupDescription: "追加の作業:"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{autoprograms}\{#AppName} 取扱説明書"; Filename: "{app}\_internal\manual\index.html"; Check: FileExists(ExpandConstant('{app}\_internal\manual\index.html'))
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{#AppName} を起動する"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; アンインストール時に残るビルド生成物 (ログ等) を掃除する。
; **利用者の設定 (~\.cw-decorder\) は消さない。** 経歴・語彙・型が入っており、
; 入れ直しのたびに消えると理不尽なため。
Type: filesandordirs; Name: "{app}\_internal\__pycache__"
