# P1, world seed 1 (2026-10-27 to 10-28; launched by Codex after the implementation audit, commit a50874f0)

25.6 M frames, 15,625 updates (12,500 at 16 x 64, then 3,125 at 4 x 4 x 256), ~9.3 h. Training log: `log.jsonl`
(every 25 updates; monitor and per-loss encoder gradient norms at each of the 10 checkpoints). The fixed monitor's G1
loss fell at every checkpoint (0.0703 -> 0.0191), so the selection rule kept checkpoint 10 (`s1/selection.json`).
S1 ran once on it through the after-P1 hook (commit ae4d66a8): `s1/results.json`, `s1/eval.log`; the frozen reader
is `s1/results.reader.pt`. `saccade-s1-world1.service` is the transient unit that ran it.
Binaries are not committed (*.pt); their sha256:
- `ckpt_01.pt`: `31d4ffe8b80e6030242560a82715e429e615503983c1dce28d9d1748a25e97d8`
- `ckpt_02.pt`: `39eb3d9d6b43104f28ce8f2acd9babea6bab4f93c2214417a97ad10f6a78dcc5`
- `ckpt_03.pt`: `5fa0fe15f35f5c807dff2751e2c920c2b8eafcb10ece6d317133f1b2896acfc5`
- `ckpt_04.pt`: `15d22c54a84d427671e02ce0429890d761ee66d4927d09db70231a3ba26e657e`
- `ckpt_05.pt`: `fe2aee881e061278096120734f982dff65d44d5719d2dbc8495884edfbfab09e`
- `ckpt_06.pt`: `d778007cdb9e8b87c2da5a0279492eb1c312b18dc2473d2b7bd5e87eb3b8bbac`
- `ckpt_07.pt`: `bdb833a12f3e1987cf3a0c29f73cdccc7a85b7ff1fb46d4cf7800c9ef731891b`
- `ckpt_08.pt`: `6f51c099d1941c2d2bce7ed00fcb6b72cdd3f9045c8e76a9e763c25a2c225de6`
- `ckpt_09.pt`: `ef46b5dc04d8064f5e7cd860ad35769567de74d514281fd27f8a897a9a46ed1a`
- `ckpt_10.pt`: `74eccc433bf5a785edd6d565edc1a970f8b468510abf2c0b07e24d28b547ebe5`
- `s1/results.reader.pt`: `f6dfaf2888ddab2108cc29dafb25c1a3920d2ceeb6f2a470dc9ef5e4291bcf58`
