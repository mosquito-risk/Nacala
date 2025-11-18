const CITATION = `
@article{guthula2025drone,
  title={Drone imagery for roof detection, classification, and segmentation to support Mosquito-borne disease risk assessment: The Nacala-Roof-Material dataset},
  author={Guthula, Venkanna Babu and Oehmcke, Stefan and Chilaule, Remigio and Zhang, Hui and Lang, Nico and Kariryaa, Ankit and Mottelson, Johan and Igel, Christian},
  journal={Science of Remote Sensing},
  pages={100306},
  year={2025},
  publisher={Elsevier}
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
