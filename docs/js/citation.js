const CITATION = `
@misc{venky2024Nacala-Roof-Material,
    title = { Nacala-Roof-Material: Exploring Multi-Modal Pretext Tasks For Geospatial Representation Learning},
    author={Venkanna Babu Guthula and Stefan Oehmcke and Remígio Chilaule and Hui Zhang and Nico Lang and Ankit Kariryaa and Johan Mottelson and Christian Igel},
    year={...},
    eprint={...},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
`

const Citation = () => {
    return (
        <section class="section hero is-white" id="BibTeX">
            <div class="hero-body">
                <div class="container is-max-desktop" data-theme="light">
                    <h2 class="title">Citation</h2>
                    Venkanna Babu Guthula, Stefan Oehmcke, Remígio Chilaule, Hui Zhang, Nico Lang, Ankit Kariryaa, Johan Mottelson & Christian Igel (2024).
                    Nacala-Roof-Material: Exploring Multi-Modal Pretext Tasks For Geospatial Representation Learning.
                    <pre><code>{CITATION}</code></pre>
                </div>
            </div>
        </section>
    )
}