const LINKS = [
    {name: "Paper", link: "", icon: "fas fa-file-pdf"},
    {name: "Code - Data", link: "https://github.com/mosquito-risk/Nacala", icon: "fab fa-github"},
    {name: "Code - Model", link: "", icon: "fab fa-github"},
    {name: "arXiv", link: "", icon: "ai ai-arxiv"}
]

const Links = () => {
    return (
        <div class="publication-links">
            {LINKS.map((link, i) => (
                <span class="link-block" key={i}>
                    <a href={link.link} target="_blank" class="external-link button is-normal is-rounded is-dark">
                        <span class="icon"><i class={link.icon}></i></span>
                        <span>{link.name}</span>
                    </a>
                </span>
            ))}
        </div>
    )
}