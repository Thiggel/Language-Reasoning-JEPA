# TextJEPA Research Reader

This folder is a zero-dependency local dashboard. It synchronizes only compact
research reports and controller status; checkpoints, datasets, source trees,
and raw run directories remain on the server.

## Install on your computer

From your computer, replace `gruenau1` with an SSH host that reaches the shared
TextJEPA filesystem:

```bash
mkdir -p ~/TextJEPA-Research-UI
rsync -a \
  laitenbf@gruenau1.informatik.hu-berlin.de:/vol/home-vol2/ml/laitenbf/TextJEPA/ui/ \
  ~/TextJEPA-Research-UI/
cd ~/TextJEPA-Research-UI
python3 server.py \
  --remote laitenbf@gruenau1.informatik.hu-berlin.de
```

The browser opens at `http://127.0.0.1:8765`. Keep the terminal open while
reading. Reports synchronize every minute. Read/unread state stays in that
browser, while marking a report as read also sends a small receipt to the
server so autonomy cannot outrun your review limit. Press **Send steering to
Codex inbox** to copy an explicit steering note to
`.researchctl/steering/inbox/<project>/` on the server.

For local testing inside the repository:

```bash
python3 ui/server.py
```

The server binds only to localhost. It does not expose a public network port.
