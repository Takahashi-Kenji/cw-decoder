; cw-key-server (打鍵サーバ) Windows インストーラ (Inno Setup 6)
;
; 無線機に電鍵線を繋いだ PC に入れる。**受信アプリとは別の配布物**である
; (運用者の指示、2026-08-31:「異なった PC で使うケースがある」)。
;
; AppId は受信アプリ・音声送出と**必ず別にすること**。同じにすると、
; 片方を入れたときにもう片方が「アップグレード」と見なされて消える。

#define AppName      "cw-key-server"
#define AppTitle     "cw-decoder 打鍵サーバ"
#define AppVersion   "0.1.0"
#define AppPublisher "cw-decoder"
#define AppExeName   "cw-key-server.exe"
#define SourceDir    "..\dist\cw-key-server"

[Setup]
AppId={{B1F4A6D2-3E77-4C55-9A18-6C2E0D5F7A91}
AppName={#AppTitle}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppTitle}
; ユーザ領域へ入れる = 管理者権限が要らない
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\dist
OutputBaseFilename=cw-key-server-{#AppVersion}-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#AppExeName}
ShowLanguageDialog=no

[Languages]
Name: "japanese"; MessagesFile: "compiler:Languages\Japanese.isl"

[Tasks]
Name: "desktopicon"; Description: "デスクトップにショートカットを作る"; GroupDescription: "追加の作業:"; Flags: unchecked
; PATH 追加は運用者の要望 (2026-08-31)。結線確認 (--check-lines) を
; どのフォルダからでも叩けるようにするため。**既定は入れない**。
Name: "addtopath"; Description: "PATH に追加する (PowerShell から cw-key-server と打てるようにする)"; GroupDescription: "追加の作業:"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppTitle}"; Filename: "{app}\{#AppExeName}"
Name: "{autodesktop}\{#AppTitle}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Registry]
; ユーザ環境変数の PATH に足す (管理者権限が要らない範囲)。
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; \
    ValueData: "{olddata};{app}"; Tasks: addtopath; \
    Check: NeedsAddPath(ExpandConstant('{app}'))

[Code]
{ 既に入っている PATH に同じ場所を二重に足さない }
function NeedsAddPath(Param: string): Boolean;
var
  OrigPath: string;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', OrigPath) then
  begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + Param + ';', ';' + OrigPath + ';') = 0;
end;
