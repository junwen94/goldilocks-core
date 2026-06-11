"""HPC submission script generator — wizard module."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

@dataclass
class _HpcEnv:
    scheduler: str | None = None          # "slurm" | "pbs" | None
    partitions: list[str] = field(default_factory=list)   # SLURM partitions
    queues: list[str] = field(default_factory=list)        # PBS queues
    cpus_per_node: dict[str, int] = field(default_factory=dict)  # partition → CPUs
    default_account: str | None = None
    # QE
    pw_x_path: str | None = None
    pw_x_version: str | None = None
    qe_modules: list[str] = field(default_factory=list)   # names from `module avail`


def _run(cmd: list[str], timeout: int = 5) -> str | None:
    """Run a command, return stdout or None on failure."""
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "COLUMNS": "200"},
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _run_shell(cmd: str, timeout: int = 5) -> str | None:
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout,
        )
        out = (r.stdout + r.stderr).strip()
        return out if out else None
    except Exception:
        return None


def detect_hpc_env() -> _HpcEnv:
    """Probe the current environment for scheduler, partition, and QE info."""
    env = _HpcEnv()

    # ── Scheduler ─────────────────────────────────────────────────────────
    if shutil.which("sbatch"):
        env.scheduler = "slurm"

        # partitions + CPUs per node
        sinfo = _run(["sinfo", "-h", "-o", "%P|%c|%D"])
        if sinfo:
            for line in sinfo.splitlines():
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    p = parts[0].rstrip("*")   # strip default marker
                    env.partitions.append(p)
                    try:
                        env.cpus_per_node[p] = int(parts[1])
                    except ValueError:
                        pass

        # default account (env var first, then sacctmgr)
        env.default_account = os.environ.get("SLURM_ACCOUNT") or os.environ.get("SBATCH_ACCOUNT")
        if not env.default_account:
            user = os.environ.get("USER", "")
            if user:
                out = _run(["sacctmgr", "show", "user", user,
                             "format=DefaultAccount", "-n", "--parsable2"])
                if out:
                    env.default_account = out.splitlines()[0].strip() or None

    elif shutil.which("qsub"):
        env.scheduler = "pbs"

        out = _run(["qstat", "-Q"])
        if out:
            for line in out.splitlines()[2:]:   # skip header lines
                parts = line.split()
                if parts:
                    env.queues.append(parts[0])

    # ── QE installation ────────────────────────────────────────────────────
    pw = shutil.which("pw.x")
    if pw:
        env.pw_x_path = pw
        ver_out = _run(["pw.x", "--version"]) or _run(["pw.x", "-v"])
        if ver_out:
            m = re.search(r"(\d+\.\d+[\.\d]*)", ver_out)
            if m:
                env.pw_x_version = m.group(1)
            else:
                env.pw_x_version = ver_out.splitlines()[0][:40]

    # QE modules (best-effort; `module` is a shell function so needs bash -c)
    mod_avail = _run_shell("module avail 2>&1 | grep -i -E '(espresso|quantum.esp|qe[/_-])'")
    if mod_avail:
        for token in re.split(r"[\s\n]+", mod_avail):
            token = token.strip("()")
            if token and not token.startswith("-") and "/" in token or token.lower().startswith(("qe", "quantum")):
                name = token.rstrip("(default)").strip()
                if name:
                    env.qe_modules.append(name)
        env.qe_modules = list(dict.fromkeys(env.qe_modules))[:8]   # dedup, cap at 8

    return env


# ---------------------------------------------------------------------------
# QE installation environment detection
# ---------------------------------------------------------------------------

@dataclass
class _InstallEnv:
    hostname: str = ""
    system_name: str = ""          # e.g. "ARCHER2", "Cirrus", "SCARF", "unknown"
    easybuild_version: str | None = None
    spack_version: str | None = None
    conda_cmd: str | None = None   # "conda" | "mamba" | None
    conda_version: str | None = None
    container_cmd: str | None = None   # "apptainer" | "singularity" | None
    gcc_version: str | None = None
    intel_version: str | None = None
    mpi_flavor: str | None = None      # "openmpi" | "intelmpi" | "mpich" | "craympi"
    mpi_version: str | None = None
    has_mkl: bool = False
    has_openblas: bool = False
    # EasyBuild: available QE easyconfigs found on the system
    eb_qe_configs: list[str] = field(default_factory=list)


_KNOWN_SYSTEMS: dict[str, str] = {
    "archer2": "ARCHER2",
    "cirrus":  "Cirrus",
    "scarf":   "SCARF",
    "jade":    "JADE2",
    "young":   "Young (Myriad)",
    "kathleen":"Kathleen",
    "thomas":  "Thomas",
    "bede":    "Bede",
    "cosma":   "COSMA",
    "hawk":    "Hawk",
    "isambard":"Isambard",
}


def detect_install_env() -> _InstallEnv:
    """Detect compilers, MPI, and package-manager tools for QE installation."""
    import socket
    env = _InstallEnv()

    env.hostname = socket.gethostname()
    hn_lower = env.hostname.lower()
    for key, name in _KNOWN_SYSTEMS.items():
        if key in hn_lower:
            env.system_name = name
            break
    else:
        env.system_name = "unknown"

    # ── Package managers ───────────────────────────────────────────────────
    if shutil.which("eb"):
        out = _run(["eb", "--version"])
        if out:
            m = re.search(r"(\d+\.\d+[\.\d]*)", out)
            env.easybuild_version = m.group(1) if m else out[:20]
        # find QE easyconfigs on this system
        eb_search = _run_shell("eb --search QuantumESPRESSO 2>/dev/null | grep -i '.eb$'")
        if eb_search:
            for line in eb_search.splitlines():
                line = line.strip()
                if line.endswith(".eb"):
                    cfg = line.split("/")[-1]
                    env.eb_qe_configs.append(cfg)
            env.eb_qe_configs = env.eb_qe_configs[:10]

    if shutil.which("spack"):
        out = _run(["spack", "--version"])
        if out:
            env.spack_version = out.splitlines()[0][:30]

    for conda in ("mamba", "conda"):
        if shutil.which(conda):
            env.conda_cmd = conda
            out = _run([conda, "--version"])
            if out:
                env.conda_version = out.strip()
            break

    for ctr in ("apptainer", "singularity"):
        if shutil.which(ctr):
            env.container_cmd = ctr
            break

    # ── Compilers ─────────────────────────────────────────────────────────
    gcc = _run(["gcc", "--version"])
    if gcc:
        m = re.search(r"(\d+\.\d+\.\d+)", gcc)
        env.gcc_version = m.group(1) if m else gcc.splitlines()[0][:20]

    for icmd in ("icc", "icx", "ifx"):
        if shutil.which(icmd):
            out = _run([icmd, "--version"])
            if out:
                m = re.search(r"(\d{4}\.\d[\.\d]*)", out)
                env.intel_version = m.group(1) if m else out[:20]
            break

    # ── MPI ───────────────────────────────────────────────────────────────
    mpirun = _run(["mpirun", "--version"]) or _run(["mpiexec", "--version"])
    if mpirun:
        lo = mpirun.lower()
        if "open mpi" in lo or "openmpi" in lo:
            env.mpi_flavor = "openmpi"
        elif "intel" in lo:
            env.mpi_flavor = "intelmpi"
        elif "mpich" in lo:
            env.mpi_flavor = "mpich"
        elif "cray" in lo:
            env.mpi_flavor = "craympi"
        m = re.search(r"(\d+\.\d+[\.\d]*)", mpirun)
        if m:
            env.mpi_version = m.group(1)

    # ── Math libraries (heuristic) ─────────────────────────────────────────
    mkl_root = os.environ.get("MKLROOT") or os.environ.get("MKL_ROOT")
    env.has_mkl = bool(mkl_root) or bool(shutil.which("mkl_link_tool"))
    env.has_openblas = bool(
        _run_shell("find /usr/lib* /usr/local/lib* $HOME/.local/lib* "
                   "-name 'libopenblas*' -maxdepth 5 2>/dev/null | head -1")
    )

    return env


# ---------------------------------------------------------------------------
# QE installation guide generator
# ---------------------------------------------------------------------------

def generate_qe_install_guide(hpc: _HpcEnv, ins: _InstallEnv) -> str:
    """Return a markdown installation guide tailored to the detected environment."""
    from datetime import date
    today = date.today().isoformat()

    lines: list[str] = []
    _w = lines.append

    _w("# QE Installation Guide")
    _w("")
    _w(f"Generated by **goldilocks** on {today}  ")
    _w(f"Host: `{ins.hostname}`  ")
    if ins.system_name != "unknown":
        _w(f"System: **{ins.system_name}**  ")
    _w("")
    _w("---")
    _w("")

    # ── Detected environment table ─────────────────────────────────────────
    _w("## Detected Environment")
    _w("")
    _w("| Item | Detected |")
    _w("|------|----------|")
    _w(f"| Scheduler | {hpc.scheduler.upper() if hpc.scheduler else '—'} |")
    _w(f"| GCC | {ins.gcc_version or '—'} |")
    _w(f"| Intel compiler | {ins.intel_version or '—'} |")
    _w(f"| MPI | {(ins.mpi_flavor or '—') + (' ' + ins.mpi_version if ins.mpi_version else '')} |")
    _w(f"| MKL | {'yes' if ins.has_mkl else '—'} |")
    _w(f"| OpenBLAS | {'yes' if ins.has_openblas else '—'} |")
    _w(f"| EasyBuild | {ins.easybuild_version or '—'} |")
    _w(f"| Spack | {ins.spack_version or '—'} |")
    _w(f"| conda/mamba | {ins.conda_cmd + ' ' + (ins.conda_version or '') if ins.conda_cmd else '—'} |")
    _w(f"| Container | {ins.container_cmd or '—'} |")
    _w("")
    _w("---")
    _w("")

    # ── Determine primary recommended path ────────────────────────────────
    if ins.easybuild_version:
        primary = "easybuild"
    elif ins.spack_version:
        primary = "spack"
    elif ins.conda_cmd:
        primary = "conda"
    elif ins.container_cmd:
        primary = "container"
    else:
        primary = "spack_install"   # install spack first, then QE

    _w("## Recommended Installation Path")
    _w("")

    if primary == "easybuild":
        _w(_eb_section(ins, is_primary=True))
    elif primary == "spack":
        _w(_spack_section(ins, is_primary=True))
    elif primary == "conda":
        _w(_conda_section(ins, is_primary=True))
    elif primary == "container":
        _w(_container_section(ins, is_primary=True))
    else:
        _w(_spack_install_first_section(ins))

    _w("")
    _w("---")
    _w("")
    _w("## Alternative Options")
    _w("")

    for alt in ["easybuild", "spack", "conda", "container"]:
        if alt != primary:
            if alt == "easybuild":
                _w(_eb_section(ins, is_primary=False))
            elif alt == "spack":
                _w(_spack_section(ins, is_primary=False))
            elif alt == "conda":
                _w(_conda_section(ins, is_primary=False))
            elif alt == "container":
                _w(_container_section(ins, is_primary=False))
            _w("")

    _w("---")
    _w("")
    _w(_verify_section())

    return "\n".join(lines)


def _eb_section(ins: _InstallEnv, is_primary: bool) -> str:
    note = "" if is_primary else "### EasyBuild\n"
    avail = ""
    if ins.eb_qe_configs:
        latest = ins.eb_qe_configs[0]
        avail = (
            "\nAvailable QE easyconfigs on this system:\n```\n"
            + "\n".join(f"  {c}" for c in ins.eb_qe_configs[:5])
            + "\n```\n"
        )
        install_cmd = f"eb {latest} --robot"
        module_name = latest.replace(".eb", "")
    else:
        install_cmd = "eb QuantumESPRESSO-7.3.1-foss-2022a.eb --robot"
        module_name = "QuantumESPRESSO/7.3.1-foss-2022a"

    archer2_tip = ""
    if ins.system_name == "ARCHER2":
        archer2_tip = (
            "\n> **ARCHER2 tip:** Load the EasyBuild module first:\n"
            "> ```bash\n> module load easybuild\n> ```\n"
            "> Pre-built easyconfigs are at `/work/y07/shared/easybuild/easyconfigs/`.\n"
        )

    return f"""{note}**EasyBuild** builds from source using the system toolchain.
Best choice when the HPC centre already maintains easyconfigs (ARCHER2, Cirrus, VSC, etc.).
{avail}{archer2_tip}
```bash
# 1. Check what QE versions are available
eb --search QuantumESPRESSO

# 2. Install (adjust version/toolchain to match what's on your system)
{install_cmd}

# 3. Load the resulting module
module load {module_name}

# 4. Verify
pw.x --version
```"""


def _spack_section(ins: _InstallEnv, is_primary: bool) -> str:
    note = "" if is_primary else "### Spack\n"
    blas = "^mkl" if ins.has_mkl else "^openblas"
    mpi_spec = ""
    if ins.mpi_flavor == "intelmpi":
        mpi_spec = " ^intel-mpi"
    elif ins.mpi_flavor == "craympi":
        mpi_spec = " ^cray-mpich"
    spec = f"quantum-espresso +mpi{mpi_spec} {blas}"

    return f"""{note}**Spack** resolves all dependencies automatically.
Users can install it in their home directory without root.

```bash
# 1. Install Spack (if not already available)
git clone --depth=1 https://github.com/spack/spack.git ~/spack
echo 'source ~/spack/share/spack/setup-env.sh' >> ~/.bashrc
source ~/spack/share/spack/setup-env.sh

# 2. Let Spack detect system compilers and libraries
spack compiler find
spack external find

# 3. Install QE
#    Spack will use detected MPI and math libraries automatically
spack install {spec}

# 4. Load
spack load quantum-espresso

# 5. Verify
pw.x --version
```

> **Tip:** Add `spack load quantum-espresso` to your job script before calling `pw.x`."""


def _conda_section(ins: _InstallEnv, is_primary: bool) -> str:
    note = "" if is_primary else "### conda-forge (quick start, not HPC-optimised)\n"
    cmd = ins.conda_cmd or "conda"
    return f"""{note}**conda-forge** provides pre-compiled QE binaries.
Easy to install, but does **not** use the system MPI or MKL — performance on large jobs will be lower.
Suitable for testing and single-node runs.

```bash
# Create a dedicated environment
{cmd} create -n qe -c conda-forge qe

# Activate
{cmd} activate qe

# Verify
pw.x --version
```

> ⚠️ For production multi-node runs, prefer EasyBuild or Spack."""


def _container_section(ins: _InstallEnv, is_primary: bool) -> str:
    note = "" if is_primary else f"### {ins.container_cmd or 'Singularity/Apptainer'} container\n"
    cmd = ins.container_cmd or "apptainer"
    return f"""{note}**Container** — pull a pre-built QE image.
Fastest to get started; performance depends on whether the container uses MPI binding.

```bash
# Pull official QE container (adjust tag for desired version)
{cmd} pull qe-7.3.sif docker://ghcr.io/quantum-espresso/qe:qe-7.3

# Run pw.x inside the container
{cmd} exec qe-7.3.sif pw.x < gl-pw-scf.in > scf.out

# For MPI-parallel runs (requires container MPI to match host MPI)
mpirun -np 16 {cmd} exec qe-7.3.sif pw.x < gl-pw-scf.in > scf.out
```

> ⚠️ MPI performance in containers depends on the host–container MPI ABI compatibility.
> Check your HPC centre's documentation for container + MPI guidance."""


def _spack_install_first_section(ins: _InstallEnv) -> str:
    return """No package manager detected. Install **Spack** first (no root required):

```bash
# Install Spack in your home directory
git clone --depth=1 https://github.com/spack/spack.git ~/spack
echo 'source ~/spack/share/spack/setup-env.sh' >> ~/.bashrc
source ~/spack/share/spack/setup-env.sh

# Detect system compilers and libraries
spack compiler find
spack external find

# Install QE
spack install quantum-espresso +mpi

# Load
spack load quantum-espresso
pw.x --version
```"""


def _verify_section() -> str:
    return """## Verification

Once installed, verify QE works:

```bash
# Check pw.x is in PATH and shows the correct version
pw.x --version

# Quick smoke test — create a minimal silicon input
cat > si_test.in << 'EOF'
&CONTROL
  calculation = 'scf'
  pseudo_dir  = './'
  outdir      = './out/'
/
&SYSTEM
  ibrav = 2, celldm(1) = 10.26, nat = 2, ntyp = 1
  ecutwfc = 30.0
/
&ELECTRONS /
ATOMIC_SPECIES
  Si  28.086  Si.upf
ATOMIC_POSITIONS alat
  Si 0.00 0.00 0.00
  Si 0.25 0.25 0.25
K_POINTS automatic
  4 4 4  0 0 0
EOF

# Run (requires Si pseudopotential — replace Si.upf with your file)
pw.x < si_test.in > si_test.out
grep "convergence" si_test.out
```
"""


def _display_env(console: Console, env: _HpcEnv) -> None:
    """Print a compact summary of what was detected."""
    t = Table(show_header=False, box=None, padding=(0, 2), show_edge=False)
    t.add_column("key", style="dim", min_width=18)
    t.add_column("value")

    # scheduler
    if env.scheduler:
        t.add_row("Scheduler", Text(env.scheduler.upper(), style="bold green"))
    else:
        t.add_row("Scheduler", Text("not detected", style="yellow"))

    # partitions / queues
    if env.partitions:
        parts_str = "  ".join(env.partitions[:10])
        t.add_row("Partitions", parts_str)
        if env.cpus_per_node:
            sample = next(iter(env.cpus_per_node.values()))
            t.add_row("CPUs / node", str(sample) + "  [dim](first partition)[/dim]")
    if env.queues:
        t.add_row("Queues", "  ".join(env.queues[:10]))

    # account
    if env.default_account:
        t.add_row("Default account", Text(env.default_account, style="bold"))

    # QE
    if env.pw_x_path:
        ver = f"  ({env.pw_x_version})" if env.pw_x_version else ""
        t.add_row("pw.x", Text(env.pw_x_path + ver, style="green"))
    else:
        t.add_row("pw.x", Text("not in PATH", style="yellow"))

    if env.qe_modules:
        t.add_row("QE module(s)", "  ".join(env.qe_modules[:4]))

    console.print(t)
    console.print()


# ---------------------------------------------------------------------------
# Main wizard
# ---------------------------------------------------------------------------

def run(console: Console) -> None:
    """Interactive HPC script generator for QE calculations."""
    console.print()
    console.rule("[bold]HPC Submission Script[/bold]", style="blue")
    console.print()

    # ── Auto-detect ────────────────────────────────────────────────────────
    with console.status("  Detecting environment…", spinner="dots"):
        env = detect_hpc_env()

    console.print("  [bold]Detected environment[/bold]")
    _display_env(console, env)

    # ── Scheduler ─────────────────────────────────────────────────────────
    if env.scheduler:
        default_sched = "1" if env.scheduler == "slurm" else "2"
    else:
        default_sched = "1"

    console.print("  Scheduler type:")
    console.print("    [bold cyan]1)[/bold cyan] SLURM  (ARCHER2, Kathleen, …)")
    console.print("    [bold cyan]2)[/bold cyan] PBS    (SCARF, older clusters)")
    sched_raw = Prompt.ask("  Choose", choices=["1", "2"], default=default_sched, show_choices=False)
    scheduler = "slurm" if sched_raw == "1" else "pbs"
    console.print()

    # ── Run directory ──────────────────────────────────────────────────────
    run_dir_str = Prompt.ask("  Run directory (with goldilocks input files)", default=".")
    run_dir = Path(run_dir_str).expanduser().resolve()
    if not run_dir.exists():
        console.print(f"  [red]Error:[/red] {run_dir} does not exist.")
        return

    in_files = sorted(run_dir.glob("gl-pw-*.in"))
    has_ph = (run_dir / "gl-ph.in").exists()

    if in_files:
        names = "  ".join(f.name for f in in_files)
        console.print(f"  [dim]Found:[/dim] {names}" + ("  gl-ph.in" if has_ph else ""))
    else:
        console.print("  [yellow]⚠[/yellow] No gl-pw-*.in files found — script will need manual editing.")
    console.print()

    # ── Job parameters ─────────────────────────────────────────────────────
    job_name = Prompt.ask("  Job name", default="goldilocks")
    nodes    = int(Prompt.ask("  Number of nodes", default="1"))

    # pick a sensible default QE module name
    default_qe_module = (
        env.qe_modules[0] if env.qe_modules
        else ("quantum_espresso" if scheduler == "slurm" else "quantum_espresso")
    )

    if scheduler == "slurm":
        default_partition = env.partitions[0] if env.partitions else "standard"
        default_cpus = str(env.cpus_per_node.get(default_partition, 128))

        ntasks_per_node = int(Prompt.ask("  MPI tasks per node", default=default_cpus))
        walltime  = Prompt.ask("  Walltime (HH:MM:SS)", default="24:00:00")

        if env.partitions:
            console.print(f"  [dim]Available partitions:[/dim] {', '.join(env.partitions)}")
        partition = Prompt.ask("  Partition", default=default_partition)

        default_acct = env.default_account or ""
        account = Prompt.ask("  Account / project code (leave blank to skip)", default=default_acct)

        qe_module = Prompt.ask("  QE module name", default=default_qe_module)

        script = _slurm_script(
            job_name=job_name,
            nodes=nodes,
            ntasks_per_node=ntasks_per_node,
            walltime=walltime,
            partition=partition,
            account=account or None,
            qe_module=qe_module,
            in_files=in_files,
            has_ph=has_ph,
        )
        script_name = "submit.sh"
    else:
        ntasks_per_node = int(Prompt.ask("  MPI tasks per node", default="16"))
        walltime  = Prompt.ask("  Walltime (HH:MM:SS)", default="24:00:00")

        if env.queues:
            console.print(f"  [dim]Available queues:[/dim] {', '.join(env.queues)}")
        queue = Prompt.ask("  Queue", default=env.queues[0] if env.queues else "standard")
        mem   = Prompt.ask("  Memory per node (e.g. 32gb)", default="32gb")

        qe_module = Prompt.ask("  QE module name", default=default_qe_module)

        script = _pbs_script(
            job_name=job_name,
            nodes=nodes,
            ntasks_per_node=ntasks_per_node,
            walltime=walltime,
            queue=queue,
            mem=mem,
            qe_module=qe_module,
            in_files=in_files,
            has_ph=has_ph,
        )
        script_name = "submit.pbs"

    # ── Write ──────────────────────────────────────────────────────────────
    out_path = run_dir / script_name
    if out_path.exists():
        if not Confirm.ask(f"  {script_name} already exists — overwrite?", default=False):
            console.print("  Skipped.")
            return

    out_path.write_text(script)
    console.print()
    console.print(f"  [green]✓[/green] Written: [bold]{out_path}[/bold]")

    if not env.pw_x_path:
        console.print("  [yellow]⚠[/yellow]  pw.x not in PATH — verify the module name in the script.")
        console.print()
        if Confirm.ask("  pw.x not found — generate a QE installation guide?", default=True):
            _do_install_guide(console, env, run_dir)

    console.print()


def _do_install_guide(console: Console, hpc_env: _HpcEnv, run_dir: Path) -> None:
    """Detect install tools, generate guide, write to file."""
    with console.status("  Detecting compilers, MPI, package managers…", spinner="dots"):
        ins = detect_install_env()

    guide = generate_qe_install_guide(hpc_env, ins)

    out_path = run_dir / "qe_install_guide.md"
    out_path.write_text(guide)
    console.print(f"  [green]✓[/green] Guide written: [bold]{out_path}[/bold]")

    # quick summary of what was detected for install
    items: list[str] = []
    if ins.easybuild_version:
        items.append(f"EasyBuild {ins.easybuild_version} → [bold]primary path[/bold]")
    if ins.spack_version:
        items.append(f"Spack {ins.spack_version}")
    if ins.conda_cmd:
        items.append(f"{ins.conda_cmd}")
    if ins.container_cmd:
        items.append(ins.container_cmd)
    if not items:
        items.append("no package manager found — guide recommends installing Spack first")
    console.print(f"  [dim]Detected:[/dim] {', '.join(items)}")


# ---------------------------------------------------------------------------
# Script templates
# ---------------------------------------------------------------------------

def _run_commands(in_files: list[Path], ntasks: int, has_ph: bool, runner: str) -> str:
    lines: list[str] = []
    if in_files:
        for f in in_files:
            out = f.stem + ".out"
            lines.append(f"{runner} -n {ntasks} pw.x < {f.name} > {out}")
        if has_ph:
            lines.append(f"{runner} -n {ntasks} ph.x < gl-ph.in > ph.out")
    else:
        lines.append("# TODO: add your pw.x run command here")
        lines.append(f"# {runner} -n {ntasks} pw.x < gl-pw-scf.in > scf.out")
    return "\n".join(lines)


def _slurm_script(
    job_name: str,
    nodes: int,
    ntasks_per_node: int,
    walltime: str,
    partition: str,
    account: str | None,
    qe_module: str,
    in_files: list[Path],
    has_ph: bool,
) -> str:
    ntasks = nodes * ntasks_per_node
    acct_line = f"#SBATCH --account={account}" if account else "# #SBATCH --account=<your-project>"
    run_cmds = _run_commands(in_files, ntasks, has_ph,
                             "srun --distribution=block:block --hint=nomultithread")
    return f"""\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --cpus-per-task=1
#SBATCH --time={walltime}
#SBATCH --partition={partition}
{acct_line}
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

module load {qe_module}

export OMP_NUM_THREADS=1

{run_cmds}
"""


def _pbs_script(
    job_name: str,
    nodes: int,
    ntasks_per_node: int,
    walltime: str,
    queue: str,
    mem: str,
    qe_module: str,
    in_files: list[Path],
    has_ph: bool,
) -> str:
    ntasks = nodes * ntasks_per_node
    run_cmds = _run_commands(in_files, ntasks, has_ph, "mpirun -np")
    return f"""\
#!/bin/bash
#PBS -N {job_name}
#PBS -l nodes={nodes}:ppn={ntasks_per_node}
#PBS -l walltime={walltime}
#PBS -l mem={mem}
#PBS -q {queue}
#PBS -j oe
#PBS -o {job_name}.out

cd $PBS_O_WORKDIR
module load {qe_module}

{run_cmds}
"""
