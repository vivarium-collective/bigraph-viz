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
</body>
</html>
