from .controller.controller import bp

def routes_list(app):
    app.register_blueprint(bp)
    return app