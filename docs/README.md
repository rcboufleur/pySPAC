# pySPAC Documentation

Complete documentation for pySPAC (python Solar Phase curve Analysis and Characterization) module.

## Files

- `mkdocs.yml` - MkDocs configuration file
- `index.md` - Documentation home page
- `getting-started.md` - How to start (installation and basic workflow)
- `setup-objects.md` - How to set up objects (PhaseCurve constructor)
- `fitting-models.md` - How to fit different models (fitModel method)
- `boundaries.md` - How to set up boundaries (parameter constraints)
- `models-methods.md` - What models are available for what fitting methods
- `uncertainties.md` - How to compute Monte Carlo uncertainties and use different percentiles
- `save-load.md` - How to save and load results (JSON serialization)
- `plotting.md` - How to plot results (matplotlib integration)
- `generate-models.md` - How to generate models from existing parameters

## Building Documentation

### Install MkDocs

```bash
pip install mkdocs mkdocs-material
```

### Serve Locally

```bash
cd pyspac_docs
mkdocs serve
```

View at: http://127.0.0.1:8000

### Build Static Site

```bash
mkdocs build
```

Creates `site/` directory with HTML files.

## Documentation Features

- Clear, direct language without bloated content
- Scientifically accurate information
- Complete coverage of all requested topics
- Working code examples throughout
- MkDocs compatible markdown format
- Ready for download and immediate use
