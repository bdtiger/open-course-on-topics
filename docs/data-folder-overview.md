# Data Folder Overview (`data-from-chiara/`)

Inspected 2026-09-16 by loading each `.h5ad` file with the anndata library and
printing its shape, metadata columns, value counts and a sample of matrix values.

**Rule for this document:** every statement below is either read directly from the
files or marked as a question. Nothing about the experiment's origin, organism,
protocol or purpose is known from the folder itself. There is no README, no
protocol and no metadata document in the folder.

## Folder layout

```
data-from-chiara/
├── h5ad/       # seven .h5ad files (AnnData objects)
└── Scripts/
    ├── ss2/    # data_process.ipynb, data_analysis.ipynb (file dates: Aug 2022)
    └── ss3/    # 2025_Scanpy_from_counts_matrix_to_adata_SS3.ipynb
                # 2025_Integration_of_SS2_and_SS3_v1.ipynb (+ .html export)
```

The notebooks in `Scripts/` have **not** been read yet. They are the only place in
the folder that could explain where the data came from.

## What an .h5ad file contains

- `.X`      matrix, rows × columns
- `.obs`    table with one row per matrix row (rows are called cells in the files)
- `.var`    table with one row per matrix column (columns are called genes in the files)
- `.raw`    optional earlier copy of `.X` and `.var` with more columns
- `.layers` other matrices with the same shape as `.X`
- `.obsm`   per-row arrays with a different number of columns (e.g. `X_pca`, `X_umap`)
- `.uns`    free-form dictionary

## The seven files

| File | rows × columns | Observed content |
|---|---|---|
| `ss2_Feifei_adata.filtered.h5ad` | 1425 × 38178 | `.X` integer values. `obs` has read-mapping stats, `sortdate` (2019_09_18, 2020_06_24), `doublet_info` all False |
| `ss3_adata_batch3_4_gfp.h5ad` | 3330 × 46891 | `.X` integer values. Has `unspliced` layer. No `uns`, no embeddings |
| `ss3_batches_3_4_gfp_unspliced_filtered_doublet_removed_v1.h5ad` | 2550 × 30680 | `.X` integer values. Has QC columns, PCA, UMAP |
| `SS2_normalized_v2.h5ad` | 1425 × 38178 | `.X` integer values despite the file name. `uns` has `log1p`. `obsm` has `Scanorama` |
| `SS3_normalized_v2.h5ad` | 2550 × 30680 | `.X` non-integer, max ≈ 6.9, row sum ≈ 4049 (consistent with log1p of normalised counts) |
| `adata_final.h5ad` | 3975 × 6176 | Both sets combined (`obs['batch']` = ss2 / ss3). Has `cell_type`, leiden clusterings, `.raw` with 22600 columns, `spliced` / `unspliced` layers |
| `adata_for_spliced_comparison.h5ad` | 3975 × 6176 | Same `obs` as `adata_final` plus columns with suffix `_velocyto` |

Row counts match across files: 1425 (ss2) + 2550 (ss3) = 3975 (`adata_final`).

## `adata_final.h5ad` in detail

### `obs['ischemia']` values and counts

| Healthy | D7 | D14 | D21 |
|---|---|---|---|
| 606 | 910 | 992 | 1467 |

### `obs['ischemia']` × `obs['batch']`

| batch | D7 | D14 | D21 | Healthy |
|---|---|---|---|---|
| ss3 | 910 | 992 | 367 | 281 |
| ss2 | 0 | 0 | 1100 | 325 |

D7 and D14 rows exist only in the ss3 batch.

### `obs['cell_type']` × `obs['ischemia']`

| cell_type | D7 | D14 | D21 | Healthy | total |
|---|---|---|---|---|---|
| Macrophages | 417 | 513 | 783 | 147 | 1860 |
| FAPs | 162 | 151 | 252 | 167 | 732 |
| APCs | 154 | 125 | 210 | 107 | 596 |
| Mural cells | 88 | 79 | 118 | 30 | 315 |
| Tenocytes | 14 | 52 | 66 | 44 | 176 |
| Endothelial cells | 14 | 14 | 22 | 67 | 117 |
| T-cells | 24 | 36 | 16 | 4 | 80 |
| Monocytes | 18 | 20 | 0 | 39 | 77 |
| Unknown | 19 | 2 | 0 | 1 | 22 |

These labels were assigned by whoever made the file. How they were assigned is not
recorded in the file; it may be in the integration notebook.

Other clustering columns present: `leiden_0.2`, `leiden_0.4`, `leiden_0.6`,
`leiden_0.8`, `leiden_1.0`, `cluster_ct`.

### `obs['cell_identity']` values

- `tdTomato+/CX3CR1+` (1715)
- `tdtomato-GFP+/CX3CR1-PdgfrB+` (835)
- `tdtomato-_GFP+` (540)
- `tdtomato+_GFP+minus` (536)
- `tdtomato+_GFP+` (349)

Related columns: `tdtomato` (plus / minus), `GFP` (plus / minus / plusminus),
`gfp_expression`, `tdtomato_expression` (numeric).

### Other `obs` columns present

`well`, `plate`, `sex` (female / male), `sorting_batch` (batch 1–4),
`S_score`, `G2M_score`, `total_counts`, `n_genes`, `pct_counts_mt`,
`pct_counts_ribo`, `pct_counts_hb`, `pct_counts_ERCC`, `percent_chrY`, `XIST-counts`,
`counts_spliced`, `counts_unspliced`, `fractions_spliced`, `fractions_unspliced`.

### Column (gene) names checked

Checked whether these names exist in `var_names`:
Pdgfrb, Pdgfra, Acta2, Col1a1, Ptprc, Pecam1, Cspg4, Rgs5, Kcnj8, Des, Adgre1, Cx3cr1, Lyz2.

- In `adata_final.X` (6176 columns): only Pecam1, Cspg4, Des are present.
- In `adata_final.raw` (22600 columns): **all of them are present, including Pdgfrb.**
- In every earlier file (30680 to 46891 columns): all present.
- No column named `Gfp`, `GFP` or `eGFP` in any file.

Mean value of the Pdgfrb column in `adata_final.raw.X`, grouped by `cell_type`:

| cell_type | mean |
|---|---|
| Mural cells | 3.26 |
| FAPs | 2.46 |
| Tenocytes | 2.46 |
| Endothelial cells | 1.11 |
| Monocytes | 0.12 |
| Macrophages | 0.11 |
| APCs | 0.03 |
| T-cells | 0.02 |
| Unknown | 0.00 |

### Matrix values in `adata_final`

| Where | Observed values | Shape |
|---|---|---|
| `.X` | non-integer, range −1.3 to exactly 10.0, row sums negative | 3975 × 6176 |
| `layers['raw']` | identical range to `.X` | 3975 × 6176 |
| `.raw.X` | non-integer, ≥ 0, max ≈ 2.1 in sample; `expm1` row sums ≈ 3774 and 4435 | 3975 × 22600 |
| `layers['spliced']`, `layers['unspliced']` | not inspected in detail | 3975 × 6176 |

Recorded processing hints in `uns`: `log1p` (empty dict), `hvg = {'flavor': 'seurat'}`,
`pca`, `neighbors`, `umap`, `wilcoxon`. `obsm` has `X_pca`, `X_umap`, `Scanorama`.
`var` has `highly_variable`, `means`, `dispersions`, `mean`, `std`.

What was actually done to produce each matrix is **not** recorded in the file
beyond these keys. The integration notebook should say.

## How to load

```python
import anndata as ad
adata = ad.read_h5ad("data-from-chiara/h5ad/adata_final.h5ad")
adata.obs["ischemia"].value_counts()
full = adata.raw.to_adata()          # 3975 × 22600
full[:, "Pdgfrb"].X
```

The project `.venv` does not have anndata: `pip install anndata`.

## Questions to ask Chiara / supervisor

Nothing below is answered by the folder.

1. What experiment is this? What organism, tissue and treatment? What do
   `ischemia`, `HLI`, D7 / D14 / D21 and Healthy refer to?
2. What are tdTomato and GFP marking here, and what were the sorting gates?
3. What do SS2 and SS3 stand for, and why do D7 and D14 exist only in ss3?
4. How were the `cell_type` labels assigned, and what do FAPs and APCs stand for?
5. Which matrix is intended for downstream modelling: `.raw.X`, `layers['spliced']`,
   or something else? What normalisation produced `.raw.X`?
6. Why does `SS2_normalized_v2.h5ad` contain integer values in `.X`?
7. Is `adata_final.h5ad` the file the previous student used for the HMM?
8. Who are Feifei and Chiara in relation to this data, and is there a protocol or
   paper describing it?

## Next step before Thursday

Read `Scripts/ss3/2025_Integration_of_SS2_and_SS3_v1.ipynb` (or its `.html` export)
and `Scripts/ss3/2025_Scanpy_from_counts_matrix_to_adata_SS3.ipynb`. They are the
only in-folder source for how the files were built.
