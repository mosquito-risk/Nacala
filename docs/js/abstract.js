const Abstract = () => {
    return (
        <section class="section hero is-light">
            <div class="container is-max-desktop">
                <div class="columns is-centered has-text-centered">
                    <div class="column is-four-fifths">
                        <h2 class="title is-3">Abstract</h2>
                        <div class="content has-text-justified">
                            <p>
                                As low-quality housing and in particular certain roof characteristics are associated
                                with an increased risk of malaria, classification of roof types based on remote
                                sensing imagery can support the assessment of malaria risk and thereby help
                                prevent the disease. To support research in this area, we release the Nacala-Roof-
                                Material dataset, which contains high-resolution drone images from Mozambique
                                with corresponding labels delineating houses and specifying their roof types. The
                                dataset defines a multi-task computer vision problem, comprising object detection,
                                classification, and segmentation. In addition, we benchmarked various state-of-the-
                                art approaches on the dataset. Canonical U-Nets, YOLOv8, and a custom decoder
                                on pretrained DINOv2 served as baselines. We show that each of the methods has
                                its advantages but none is superior on all tasks, which highlights the potential of
                                our dataset for future research in multi-task learning. While the tasks are closely
                                related, accurate segmentation of objects does not necessarily imply accurate
                                instance separation, and vice versa. We address this general issue by introducing a
                                variant of the deep ordinal watershed (DOW) approach that additionally separates
                                the interior of objects, allowing for improved object delineation and separation. We
                                show that our DOW variant is a generic approach that improves the performance of
                                both U-Net and DINOv2 backbones, leading to a better trade-off between semantic
                                segmentation and instance segmentation.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    )
}
