const CITATION = `
@misc{guthula2024nacalaroofmaterial,
    title={Nacala-Roof-Material: Drone Imagery for Roof Detection, Classification, and Segmentation to Support Mosquito-borne Disease Risk Assessment}, 
    author={Venkanna Babu Guthula and Stefan Oehmcke and Remigio Chilaule and Hui Zhang and Nico Lang and Ankit Kariryaa and Johan Mottelson and Christian Igel},
    year={2024},
    eprint={2406.04949},
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
                    <br />
                    <br />
                    <pre><code>{CITATION}</code></pre>
                </div>
            </div>
        </section>
    )
}