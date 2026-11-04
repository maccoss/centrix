; -----------------------------------------------------------------------------
; Inno Setup 6 script for Centrix (Windows x64).
;
; Built by ..\package.ps1 -Setup, which publishes a self-contained win-x64 Centrix into a
; staging folder and runs ISCC with:
;   /DStagingDir=<the staged folder>  /DMyAppVersion=<version>
;   /DOutputDir=<dist>  /DOutputBaseFilename=Centrix-Setup-<version>
;
; The payload carries its own .NET runtime, so nothing else needs to be installed. The
; installer asks whether to install for the current user (no administrator rights, under
; %LOCALAPPDATA%\Programs) or for everyone (under %ProgramFiles%), offers to put centrix on
; the PATH so it runs from any terminal, and uninstalls through Apps & features, taking the
; PATH entry back off.
; -----------------------------------------------------------------------------

#define MyAppName "Centrix"
#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif
#define MyAppPublisher "MacCoss Lab, University of Washington"
#define MyAppURL "https://github.com/maccoss/centrix"

#ifndef StagingDir
  #error StagingDir must be passed (/DStagingDir=<published folder>); see package.ps1
#endif
#ifndef OutputDir
  #define OutputDir "..\dist"
#endif
#ifndef OutputBaseFilename
  #define OutputBaseFilename "Centrix-Setup-" + MyAppVersion
#endif

[Setup]
AppId={{5B0E3A51-6C2F-4E0B-9C5E-2E8E1B7F6C11}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
VersionInfoVersion={#MyAppVersion}
DefaultDirName={autopf}\{#MyAppName}
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\centrix.exe
OutputDir={#OutputDir}
OutputBaseFilename={#OutputBaseFilename}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
; Per-user by default, with the choice of a per-machine install offered in a dialog.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
ChangesEnvironment=yes
LicenseFile={#StagingDir}\LICENSE

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "addtopath"; Description: "Add centrix to the PATH, so it runs from any terminal"; GroupDescription: "Command line:"

[Files]
Source: "{#StagingDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Registry]
; The PATH entry, in the user's environment or the machine's, as the install is per-user or
; per-machine. Appended only when absent; the uninstaller removes it (see [Code]).
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; \
    Check: IsAdminInstallMode = False and NeedsAddPath(ExpandConstant('{app}')); Tasks: addtopath
Root: HKLM; Subkey: "SYSTEM\CurrentControlSet\Control\Session Manager\Environment"; ValueType: expandsz; \
    ValueName: "Path"; ValueData: "{olddata};{app}"; \
    Check: IsAdminInstallMode and NeedsAddPath(ExpandConstant('{app}')); Tasks: addtopath

[Code]
function PathKey(var Root: Integer): String;
begin
  if IsAdminInstallMode then
  begin
    Root := HKEY_LOCAL_MACHINE;
    Result := 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment';
  end
  else
  begin
    Root := HKEY_CURRENT_USER;
    Result := 'Environment';
  end;
end;

function NeedsAddPath(Dir: String): Boolean;
var
  Root: Integer;
  Key, Path: String;
begin
  Key := PathKey(Root);
  if not RegQueryStringValue(Root, Key, 'Path', Path) then
  begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + Uppercase(Dir) + ';', ';' + Uppercase(Path) + ';') = 0;
end;

{ Removes the install directory from the PATH as a whole ';'-delimited entry, wherever it is (the first entry too),
  never as a prefix of a longer one (C:\Centrix of C:\CentrixTools). }
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Root: Integer;
  Key, Path, Dir, Padded: String;
  P: Integer;
begin
  if CurUninstallStep <> usPostUninstall then
    exit;
  Key := PathKey(Root);
  if not RegQueryStringValue(Root, Key, 'Path', Path) then
    exit;
  Dir := ExpandConstant('{app}');
  Padded := ';' + Path + ';';
  P := Pos(';' + Uppercase(Dir) + ';', Uppercase(Padded));
  if P = 0 then
    exit;
  { Drop the entry and the ';' before it; the padding keeps one ';' between its neighbors. }
  Delete(Padded, P, Length(Dir) + 1);
  Path := Copy(Padded, 2, Length(Padded) - 2);
  RegWriteExpandStringValue(Root, Key, 'Path', Path);
end;
