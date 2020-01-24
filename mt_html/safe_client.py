import boto3

class SafeClient:
    """
    A safer version of boto3's client, which only exposes functions
    which do not involve budget actions.
    """
    def __init__(self, **args):
        """
        Initialize the internal boto3 client.
        Takes the arguemnts needed to init the client
        """
        self._client = boto3.client(**args)

    def list_qualification_types(self, **args):
        """
        Boto3 wrapper.
        """
        return self._client.list_qualification_types(**args)

    def create_qualification_type(self, **args):
        """
        Boto3 wrapper.
        """
        return self._client.create_qualification_type(**args)

    def accept_qualification_request(self, **args):
        """
        Boto3 wrapper.
        """
        return self._client.accept_qualification_request(**args)

    def list_qualification_requests(self, **args):
        """
        Boto3 wrapper.
        """
        return self._client.list_qualification_requests(**args)

    def reject_qualification_request(self, **args):
        """
        Boto3 wrapper.
        """
        return self._client.reject_qualification_request(**args)

    def notify_workers(self, **args):
        """
        Boto3 wrapper.
        """
        return self._client.notify_workers(**args)
