const Models = () => {
    return (
        <div>
            <section class="section hero is-light">
                <div class="container is-max-desktop">
                    <div class="columns is-centered has-text-centered">
                        <div class="column is-four-fifths">
                            <h2 class="title is-3">Benchmarked Methods</h2>
                            <div class="content has-text-justified">
                                <ul>
                                    <li>As baselines, we
                                        considered a U-Net (Ronneberger et al., 2015), YOLOv8 (Jocher et al., 2023), and a model based on
                                        DINOv2 (Oquab et al., 2024).
                                    </li>
                                    <li>urthermore, we considered a novel U-Net variant based on the deep
                                        ordinal watershed method (Cheng et al., 2024).
                                    </li>
                                    <li>These approaches are compared in two settings.
                                        <ul>
                                            <li><b>Two-stage setting:</b> we first solved the building segmentation and separation tasks and afterwards
                                                classified the roof material for each detected building. </li>
                                            <li><b>End-to-end setting:</b> segmentation and
                                                classification were done in parallel. </li>
                                        </ul>

                                    </li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </section>
            <section class="section hero is-light">
                <figure class="model-image">
                    <img src="images/heads2.png" alt="model" />
                    <figcaption style={{ width: "fitContent", height: "fitContent" }}>
                        Figure 2: U-NetDOW architecture producing two output maps, segmenting objects and their interiors, respectively. <br />The architecture differs from the baseline U-Net only in the output heads.
                    </figcaption>
                </figure>
                <br />
                <figure class="model-image">
                    <img src="images/dinovc.png" alt="model" />
                    <figcaption style={{ width: "fitContent", height: "fitContent" }}>
                    Figure 3: The architecture of the DINOv2 based roof material classifier. <br />A classifier (e.g., logistic
regression) is applied to the resulting feature vector. 
                    </figcaption>
                </figure>
            </section>
        </div>
    )
}
