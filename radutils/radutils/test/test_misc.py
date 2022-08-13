import unittest

import radutils.misc as rad_misc


class MiscUtilsTestCase(unittest.TestCase):
    def test_generate_session_id(self):
        session_id = rad_misc.create_or_resume_session()
        session_split = session_id.split("-")
        self.assertEqual(len(session_split), 4)

    def test_parse_session_id(self):
        session_id_full = "03-01T11:23:24-d894a7-d507c-Baseline-LSTM--2"
        parse_id_full = rad_misc.parse_session_name(session_id_full)
        self.assertEqual(parse_id_full, ("03-01T11:23:24", "d894a7", "d507c", "Baseline-LSTM", 2))

        session_id_no_param_level = "03-01T11:23:24-d894a7-d507c-Baseline-LSTM"
        parse_id_no_param = rad_misc.parse_session_name(session_id_no_param_level)
        self.assertEqual(parse_id_no_param, ("03-01T11:23:24", "d894a7", "d507c", "Baseline-LSTM", None))

        session_id_no_session_name = "03-01T11:23:24-d894a7-d507c"
        parse_id_no_name = rad_misc.parse_session_name(session_id_no_session_name)
        self.assertEqual(parse_id_no_name, ("03-01T11:23:24", "d894a7", "d507c", None, None))

        session_id_no_session_name_with_param = "03-01T11:23:24-d894a7-d507c--2"
        parse_id_no_name_with_param = rad_misc.parse_session_name(session_id_no_session_name_with_param)
        self.assertEqual(parse_id_no_name_with_param, ("03-01T11:23:24", "d894a7", "d507c", None, 2))


if __name__ == "__main__":
    unittest.main()
