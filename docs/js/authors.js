const AUTHORS = [
    {
        name: "Venkanna Babu Guthula",
        affiliation: [1],
        link: "https://scholar.google.com/citations?user=LY1bSd8AAAAJ&hl=en&oi=ao",
    },
    {
        name: "Stefan Oehmcke",
        affiliation: [2],
        link: "https://scholar.google.de/citations?hl=en&user=k9EWJmcAAAAJ&view_op=list_works",
    },
    {
        name: "Remígio Chilaule",
        affiliation: [3, 4],
        link: "https://scholar.google.com/citations?user=p58Vlu4AAAAJ&hl=en&oi=ao",
    },
    {
        name: "Hui Zhang",
        affiliation: [1],
        link: "https://scholar.google.com/citations?user=nOSslqEAAAAJ&hl=en",
    },
    {
        name: "Nico Lang",
        affiliation: [1],
        link: "https://langnico.github.io/",
    },
    {
        name: "Ankit Kariryaa",
        affiliation: [1],
        link: "https://ankitkariryaa.github.io/",
    },
    {
        name: "Johan Mottelson",
        affiliation: [3],
        link: "http://mottelson.com/",
    },
    {
        name: "Christian Igel",
        affiliation: [1],
        link: "https://christian-igel.github.io/",
    },
];


const Authors = () => {
    return (
        <div class="is-size-5 publication-authors">
            <span class="author-block">
                {AUTHORS.map((author, i) => (
                    <span key={i}>
                        <a href={author.link} target="_blank" style={{ color: '#3273dc' }}>
                            {author.name}
                        </a>
                        {author.affiliation.map((affiliation, j) => (
                            <span>
                                <sup key={j}>{affiliation}</sup>
                                <sup>{j < author.affiliation.length -1 ? "," : ""}</sup>
                            </span>
                        ))}
                        {/* <sup>{author.affiliation}</sup> */}
                        {i < AUTHORS.length - 1 ? ", " : ""}
                    </span>
                ))}
            </span>
            <div class="is-size-5 publication-authors">
                <span className="author-block">
                    <sup>1</sup>
                    <a href="https://di.ku.dk/english/" style={{ color: '#3273dc' }}>University of Copenhagen</a>,&nbsp;
                </span>
                <span className="author-block">
                    <sup>2</sup>
                    <a href="https://www.uni-rostock.de/en/" style={{ color: '#3273dc' }}>University of Rostock</a>,&nbsp;
                </span>
                <span className="author-block">
                    <sup>3</sup>
                    <a href="https://royaldanishacademy.com/en" style={{ color: '#3273dc' }}>Royal Danish Academy</a>,&nbsp;
                </span>
                <span className="author-block">
                    <sup>4</sup>
                    <a href="https://mapeandomeubairro.org/" style={{ color: '#3273dc' }}>#MapeandoMeuBairro</a>
                </span>
            </div>
        </div>
    )
}
