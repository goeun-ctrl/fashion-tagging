"""Materialize pinned common code into a NEW directory without switching a branch."""
from pathlib import Path
import argparse,hashlib,io,json,shutil,subprocess,tarfile
COMMIT='c5827dfa749e276d4e5431628a06730b26c4841f'
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--git',default='git');a=p.parse_args()
    if a.out.exists():raise FileExistsError('Output exists; preserve it and choose a new directory')
    result=subprocess.run([a.git,'-C',str(a.repo),'cat-file','-e',COMMIT+'^{commit}'],capture_output=True)
    if result.returncode:subprocess.run([a.git,'-C',str(a.repo),'fetch','--depth','1','origin',COMMIT],check=True)
    raw=subprocess.check_output([a.git,'-C',str(a.repo),'archive','--format=tar',COMMIT,'common'])
    a.out.mkdir(parents=True,exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(raw)) as t:
        for entry in t.getmembers():
            target=(a.out/entry.name).resolve()
            assert target.is_relative_to(a.out.resolve()) and not entry.issym() and not entry.islnk()
            if entry.isdir():target.mkdir(parents=True,exist_ok=True)
            elif entry.isfile():
                target.parent.mkdir(parents=True,exist_ok=True)
                with t.extractfile(entry) as src:raw=src.read()
                # The running reference uses LF; the original Git commit stores CRLF text.
                if target.suffix.lower() in {'.py','.txt','.csv','.yaml','.yml','.md','.json','.toml'}:raw=raw.replace(b'\r\n',b'\n')
                with target.open('xb') as dst:dst.write(raw)
            else:raise ValueError('Unsupported archive entry')
    files={f.relative_to(a.out).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.out.rglob('*')) if f.is_file()}
    (a.out/'REFERENCE_PROVENANCE.json').write_text(json.dumps(dict(commit=COMMIT,text_line_endings='LF to match the live server reference',sha256=files),indent=2),encoding='utf-8')
    print('Pinned common reference ready:',a.out)
if __name__=='__main__':main()
