<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{{ resources['metadata']['name'] }}</title>
    <style>
        /* Add your custom CSS styles here */
    </style>
</head>
<body>
    <nav id="outline">
        <h2>Outline</h2>
        <ul id="outline-list"></ul>
    </nav>
    {% for cell in nb.cells %}
        {% if cell.cell_type == 'markdown' %}
            <div class="markdown">
                {{ cell.source | markdown2html }}
            </div>
        {% elif cell.cell_type == 'code' %}
            <div class="code">
                <pre>{{ cell.source }}</pre>
                {% if cell.outputs %}
                    {% for output in cell.outputs %}
                        {% if output.output_type == 'execute_result' or output.output_type == 'display_data' %}
                            {% if 'text/html' in output.data %}
                                {{ output.data['text/html'] | safe }}
                            {% elif 'text/plain' in output.data %}
                                <pre>{{ output.data['text/plain'] }}</pre>
                            {% endif %}
                        {% endif %}
                    {% endfor %}
                {% endif %}
            </div>
        {% endif %}
    {% endfor %}
    <script>
        function generateOutline() {
            const headers = document.querySelectorAll('.markdown h1, .markdown h2, .markdown h3, .markdown h4, .markdown h5, .markdown h6');
            const outlineList = document.getElementById('outline-list');

            headers.forEach((header, index) => {
                const linkId = `header-link-${index}`;
                header.setAttribute('id', linkId);

                const listItem = document.createElement('li');
                const link = document.createElement('a');
                link.setAttribute('href', `#${linkId}`);
                link.textContent = header.textContent;

                listItem.appendChild(link);
                outlineList.appendChild(listItem);
            });
        }

        window.onload = generateOutline;
    </script>
</body>
</html>
