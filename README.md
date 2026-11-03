# Which speedrun tricks help on their own?

COMP 5923 (Research in Computer Science) project, Fall 2026. Shahbaz Singh.

Keller Jordan's modded-nanoGPT speedrun stacks lots of changes on top of Karpathy's nanoGPT.
Because they are added one after another, it's hard to tell what each one does by itself.
This project takes three of them, RoPE, RMSNorm and the Muon optimizer, and tests each one
on its own against plain nanoGPT, trained on character-level Shakespeare at matched compute
and over several seeds.
